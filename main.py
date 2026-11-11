"""photo-geolocator-01 entry: "Find where a photo was taken".

Two-pass vision geolocation with a deterministic code-only fallback.

Pass 1 asks an allowlisted vision model (claude-sonnet-5-5 on the Anthropic
route, gpt-6.1-sol on the OpenAI route) to work through a fixed evidence
checklist (signs and language, driving side and plates, road furniture,
architecture, vegetation, climate and sun) and commit to one specific place
as JSON. Pass 2 sends the photos the first pass was unsure about to the
stronger model on the same route (claude-opus-5-5 / gpt-6-sol) with the
first answer as a hypothesis to verify or correct.

Model calls go through the trial gateway on the entrant's own pushed API
key (the gateway swaps the trial key for the real one). A vendor whose
calls are refused 401 has no key pushed: the entry switches vendor once,
and if neither vendor answers, every photo is placed by the fallback: a
fixed prior point, the spherical median of the 75 practice photos shipped
in the competition kit (an aggregate statistic of the public practice
pool, computed offline; no per-photo data is carried). Every photo is
answered exactly once, in photo order.

Determinism: every request derives only from the photo bytes and the
recorded answers before it; the flow is strictly sequential, posts in
photo order, and reads no clock and no randomness.

Python 3.13, standard library only.
"""

import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request

# ---------------------------------------------------------------- config

ENV_URL = (os.environ.get("MAKEFUN_ENV_URL") or "").rstrip("/")
ANTHROPIC_URL = (os.environ.get("ANTHROPIC_BASE_URL")
                 or "https://api.anthropic.com").rstrip("/")
OPENAI_URL = (os.environ.get("OPENAI_BASE_URL")
              or "https://api.openai.com").rstrip("/")
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY") or ""
OPENAI_KEY = os.environ.get("OPENAI_API_KEY") or ""

PRIMARY_MODEL = {"anthropic": "claude-sonnet-5-5", "openai": "gpt-6.1-sol"}
EXPERT_MODEL = {"anthropic": "claude-opus-5-5", "openai": "gpt-6-sol"}
PRIMARY_MAX_TOKENS = 900
EXPERT_MAX_TOKENS = 700
CONFIDENCE_THRESHOLD = 0.55
MAX_EXPERT_PASSES = 8

# Spherical median of the kit's 75 practice photos (see module docstring).
FALLBACK_LAT = 56.395634
FALLBACK_LON = 30.658740

RETRY_STATUSES = {408, 429, 500, 502, 503, 504, 529}
RETRIES = 3
BUDGET_REFUSALS = {"budget_exhausted", "request_cap"}

CHECKLIST = (
    "Examine the photo systematically:\n"
    "- Text and signs: language, script or alphabet, business names, road-sign "
    "wording, web domains, phone-number formats.\n"
    "- Vehicles and traffic: which side of the road they drive on, license-plate "
    "shapes and colours, locally common car models.\n"
    "- Road infrastructure: lane-marking colours and patterns, sign shapes, "
    "bollards, utility poles, guardrails, pavement and kerb style.\n"
    "- Buildings and landscape: architecture, roof types, vegetation and crops, "
    "terrain, climate, and the light (sun angle and shadows).\n"
    "Commit to the single most likely location: a specific city or street area, "
    "not a regional or national average. If torn between two places, pick the "
    "more probable one."
)

PRIMARY_PROMPT = (
    "You are a world-class geolocation analyst. This street-level photo was "
    "taken somewhere on Earth. Determine where.\n\n"
    + CHECKLIST + "\n\n"
    "Answer with a brief evidence summary (at most 100 words), then a FINAL "
    "line that is only this JSON:\n"
    '{"lat": <decimal degrees, north positive>, "lon": <decimal degrees, '
    'east positive>, "confidence": <your confidence, 0.0 to 1.0>}'
)

EXPERT_PROMPT = (
    "You are a world-class geolocation analyst. Another analyst located this "
    "street-level photo at latitude {lat}, longitude {lon}. Their evidence "
    "summary: {why}\n\n"
    "Independently re-examine the photo with the same checklist:\n"
    + CHECKLIST + "\n\n"
    "Decide whether that location is right. If it is wrong or imprecise, give "
    "your corrected best estimate, again a specific place, not a regional "
    "average.\n\n"
    "Answer with a brief justification (at most 80 words), then a FINAL line "
    "that is only this JSON:\n"
    '{{"lat": <decimal degrees, north positive>, "lon": <decimal degrees, '
    'east positive>, "confidence": <your confidence, 0.0 to 1.0>}}'
)

_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


# ---------------------------------------------------------------- http

def http(method, url, body=None, headers=None, timeout=120):
    """One HTTP exchange. Returns (status, parsed_or_text_body).

    Raises urllib.error.URLError on a connection that failed before an
    answer. A status in RETRY_STATUSES is retried with fixed backoff,
    unless the response says x-should-retry: false (the baseline's rule).
    """
    data = None
    hdrs = dict(headers or {})
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        hdrs.setdefault("content-type", "application/json")
    for attempt in range(RETRIES + 1):
        req = urllib.request.Request(url, data=data, method=method,
                                     headers=hdrs)
        try:
            with _opener.open(req, timeout=timeout) as res:
                return res.status, _decode(res.read())
        except urllib.error.HTTPError as e:
            payload = _decode(e.read())
            if (e.code in RETRY_STATUSES
                    and e.headers.get("x-should-retry") != "false"
                    and attempt < RETRIES):
                time.sleep(2 * (attempt + 1))
                continue
            return e.code, payload


def _decode(raw):
    text = raw.decode("utf-8", "replace")
    try:
        return json.loads(text)
    except ValueError:
        return text


def error_type(payload):
    if isinstance(payload, dict):
        err = payload.get("error")
        if isinstance(err, dict) and isinstance(err.get("type"), str):
            return err["type"]
        if isinstance(payload.get("type"), str):
            return payload["type"]
    return ""


# ---------------------------------------------------------------- parsing

def _is_lat(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) \
        and -90.0 <= float(v) <= 90.0


def _is_lon(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) \
        and -180.0 <= float(v) <= 180.0


def parse_answer(text):
    """(lat, lon, confidence, prose) from a model answer, or None.

    Prefers the LAST JSON object carrying lat/lon (the prompt asks for the
    JSON on the final line); falls back to the first JSON object with
    coordinates, then to the first pair of in-range decimal numbers.
    """
    if not isinstance(text, str) or not text:
        return None
    objects = []
    i = 0
    while True:
        start = text.find("{", i)
        if start < 0:
            break
        end = text.find("}", start)
        if end < 0:
            break
        chunk = text[start:end + 1]
        i = end + 1
        try:
            obj = json.loads(chunk)
        except ValueError:
            continue
        if isinstance(obj, dict):
            objects.append(obj)
    for obj in reversed(objects):
        lat = obj.get("lat", obj.get("latitude"))
        lon = obj.get("lon", obj.get("lng", obj.get("longitude")))
        if _is_lat(lat) and _is_lon(lon):
            conf = obj.get("confidence")
            if not _is_lat(conf) or isinstance(conf, bool):
                conf = None
            elif float(conf) < 0.0 or float(conf) > 1.0:
                conf = min(1.0, max(0.0, float(conf) / 100.0)) \
                    if float(conf) <= 100.0 else None
            prose = text[:text.rfind("{")].strip()
            return float(lat), float(lon), \
                (None if conf is None else float(conf)), prose
    import re
    nums = [m.group(0) for m in re.finditer(r"-?\d+\.\d+", text)]
    for j in range(len(nums) - 1):
        a, b = float(nums[j]), float(nums[j + 1])
        if -90.0 <= a <= 90.0 and -180.0 <= b <= 180.0:
            return a, b, None, text.strip()
    return None


# ---------------------------------------------------------------- model calls

class Gateway:
    """State over the trial's two gateway routes.

    vendor: the route that answered OK most recently (tried first).
    dead:   routes that refused in a way retrying cannot fix this trial.
    budget_dead: a budget/request-cap refusal ends model calls for the
    rest of the trial (the baseline's rule: a refusal is final).
    """

    def __init__(self):
        self.vendor = None
        self.dead = set()
        self.budget_dead = False
        if not ANTHROPIC_KEY:
            self.dead.add("anthropic")
        if not OPENAI_KEY:
            self.dead.add("openai")

    def _anthropic(self, jpeg_b64, model, max_tokens, prompt):
        body = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image",
                     "source": {"type": "base64",
                                "media_type": "image/jpeg",
                                "data": jpeg_b64}},
                    {"type": "text", "text": prompt},
                ],
            }],
        }
        status, payload = http(
            "POST", ANTHROPIC_URL + "/v1/messages", body,
            headers={"x-api-key": ANTHROPIC_KEY,
                     "anthropic-version": "2023-06-01"},
            timeout=300)
        if status != 200:
            return ("refused", status, payload)
        parts = []
        if isinstance(payload, dict):
            for block in payload.get("content") or []:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text") or "")
        return ("ok", 200, "\n".join(parts))

    def _openai(self, jpeg_b64, model, max_tokens, prompt):
        body = {
            "model": model,
            "max_completion_tokens": max_tokens,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": "data:image/jpeg;base64,"
                                           + jpeg_b64}},
                    {"type": "text", "text": prompt},
                ],
            }],
        }
        status, payload = http(
            "POST", OPENAI_URL + "/v1/chat/completions", body,
            headers={"authorization": "Bearer " + OPENAI_KEY},
            timeout=300)
        if status != 200:
            return ("refused", status, payload)
        text = ""
        if isinstance(payload, dict):
            choices = payload.get("choices") or []
            if choices and isinstance(choices[0], dict):
                msg = choices[0].get("message") or {}
                content = msg.get("content")
                if isinstance(content, str):
                    text = content
                elif isinstance(content, list):
                    text = "\n".join(
                        p.get("text") or "" for p in content
                        if isinstance(p, dict))
        return ("ok", 200, text)

    def ask(self, jpeg, model_by_vendor, max_tokens, prompt):
        """One model call on the best live route.

        Returns (outcome, text): outcome is "ok" (text is the model's
        answer), "refused" (no route answered; text says why) — callers
        treat both refusals and no-coordinate answers via the fallback.
        """
        jpeg_b64 = base64.b64encode(jpeg).decode("ascii")
        order = ["anthropic", "openai"]
        if self.vendor == "openai":
            order.reverse()
        for vendor in order:
            if vendor in self.dead or self.budget_dead:
                continue
            caller = self._anthropic if vendor == "anthropic" \
                else self._openai
            try:
                kind, status, payload = caller(
                    jpeg_b64, model_by_vendor[vendor], max_tokens, prompt)
            except Exception as e:  # connection failed before an answer
                sys.stderr.write(
                    f"geolocator: {vendor} call failed: {e}\n")
                self.dead.add(vendor)
                continue
            if kind == "ok":
                self.vendor = vendor
                return "ok", payload
            etype = error_type(payload)
            sys.stderr.write(
                f"geolocator: {vendor} refused (HTTP {status}): "
                f"{etype or str(payload)[:160]}\n")
            if etype in BUDGET_REFUSALS:
                self.budget_dead = True
                return "refused", etype
            # 401 (no key for this vendor), model_not_allowed, a wrong
            # route, or exhausted transient retries: this route is done.
            self.dead.add(vendor)
        return "refused", "no live route"


# ---------------------------------------------------------------- the entry

def locate(gateway, jpeg, experts_left):
    """(lat, lon, experts_left) for one photo."""
    outcome, text = gateway.ask(jpeg, PRIMARY_MODEL,
                                PRIMARY_MAX_TOKENS, PRIMARY_PROMPT)
    if outcome != "ok":
        return FALLBACK_LAT, FALLBACK_LON, experts_left
    parsed = parse_answer(text)
    if parsed is None:
        return FALLBACK_LAT, FALLBACK_LON, experts_left
    lat, lon, conf, prose = parsed
    unsure = conf is None or conf < CONFIDENCE_THRESHOLD
    if unsure and experts_left > 0 and not gateway.budget_dead:
        why = (prose or "no summary given")[-400:]
        prompt = EXPERT_PROMPT.format(
            lat=f"{lat:.6f}", lon=f"{lon:.6f}", why=why)
        outcome2, text2 = gateway.ask(jpeg, EXPERT_MODEL,
                                      EXPERT_MAX_TOKENS, prompt)
        if outcome2 == "ok":
            experts_left -= 1
            parsed2 = parse_answer(text2)
            if parsed2 is not None:
                return parsed2[0], parsed2[1], experts_left
    return lat, lon, experts_left


def main():
    if not ENV_URL:
        sys.stderr.write("geolocator: MAKEFUN_ENV_URL is not set\n")
        return 1
    status, listing = http("GET", ENV_URL + "/photos", timeout=60)
    if status != 200 or not isinstance(listing, dict):
        sys.stderr.write(
            f"geolocator: cannot list photos (HTTP {status})\n")
        return 1
    photos = [p for p in listing.get("photos") or []
              if isinstance(p, dict) and not p.get("answered")]
    gateway = Gateway()
    experts_left = MAX_EXPERT_PASSES
    answers = []
    for photo in photos:
        pid = photo.get("id")
        url = photo.get("url") or f"/photos/{pid}.jpg"
        lat, lon = FALLBACK_LAT, FALLBACK_LON
        jpeg = fetch_bytes(ENV_URL + url)
        if jpeg is not None:
            try:
                lat, lon, experts_left = locate(gateway, jpeg, experts_left)
            except Exception as e:
                sys.stderr.write(
                    f"geolocator: {pid}: locate failed: {e}\n")
                lat, lon = FALLBACK_LAT, FALLBACK_LON
        answers.append({
            "id": pid,
            "lat": round(max(-90.0, min(90.0, lat)), 6),
            "lon": round(max(-180.0, min(180.0, lon)), 6),
        })
    if not answers:
        sys.stderr.write("geolocator: no photos to answer\n")
        return 0
    status, payload = http("POST", ENV_URL + "/answers",
                           {"answers": answers}, timeout=120)
    sys.stdout.write(
        f"geolocator: posted {len(answers)} answers: "
        f"HTTP {status} {payload}\n")
    return 0 if status == 200 else 1


def fetch_bytes(url):
    """GET a URL's raw body (for the JPEGs), or None."""
    for attempt in range(RETRIES + 1):
        req = urllib.request.Request(url, method="GET")
        try:
            with _opener.open(req, timeout=60) as res:
                return res.read()
        except urllib.error.HTTPError as e:
            if e.code in RETRY_STATUSES and attempt < RETRIES:
                time.sleep(2 * (attempt + 1))
                continue
            sys.stderr.write(
                f"geolocator: GET {url} -> HTTP {e.code}\n")
            return None
        except Exception as e:
            sys.stderr.write(f"geolocator: GET {url} failed: {e}\n")
            return None
    return None


if __name__ == "__main__":
    sys.exit(main())
