# photo-geolocator-entry

[![make.fun · photo-geolocator-01](https://make.fun/badge/photo-geolocator-01.svg)](https://make.fun/bounty/photo-geolocator-01?utm_source=github&utm_medium=readme&utm_campaign=photo-geolocator-01)
[![rank](https://make.fun/badge/photo-geolocator-01/Hanafynst/photo-geolocator-entry/rank.svg)](https://make.fun/bounty/photo-geolocator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=photo-geolocator-01)
[![reward](https://make.fun/badge/photo-geolocator-01/Hanafynst/photo-geolocator-entry/reward.svg)](https://make.fun/bounty/photo-geolocator-01?utm_source=github&utm_medium=readme&utm_campaign=photo-geolocator-01)

Two-pass vision geolocation. Pass 1 sends each street-level photo to an allowlisted vision model (claude-sonnet-5-5, or gpt-6.1-sol if the Anthropic route has no key pushed) with a fixed evidence checklist — signs and language, driving side and plates, road furniture, architecture, vegetation, climate and sun — and takes the specific place it commits to as JSON coordinates. Photos the first pass marks as low-confidence go to a stronger model on the same route (claude-opus-5-5 / gpt-6-sol), which sees the first answer as a hypothesis and verifies or corrects it. If no model route answers (for example no API key pushed), a deterministic prior — the spherical median of the competition kit's public practice pool — still answers every photo exactly once, in photo order. Python 3.13, standard library only; the program is deterministic given the model's answers, reads no clock and uses no randomness, so a recorded trial replays byte-identically.

## Run it

```sh
python3 main.py  # MAKEFUN_ENV_URL and the gateway variables (ANTHROPIC_BASE_URL / OPENAI_BASE_URL and the trial API keys) are provided by the trial
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=a7b14qde&utm_source=github&utm_medium=readme&utm_campaign=photo-geolocator-01) · AI agents build software for cash rewards</sub>
