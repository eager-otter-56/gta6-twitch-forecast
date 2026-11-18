# Launch-Day Forecaster

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/eager-otter-56/gta6-twitch-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/eager-otter-56/gta6-twitch-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Entry for the make.fun competition
[gta6-twitch-ccu-forecast-01](https://make.fun/bounty/gta6-twitch-ccu-forecast-01):
forecast Grand Theft Auto VI's launch-day Twitch viewers, hour by hour
(hours 2–24 UTC, five quantiles each), from the first hour's print alone.

## Model

`main.py` (Python 3.13, standard library only) rebuilds every comparable
launch's release-day curve from the case file itself, takes the per-hour
log ratios `log(v(h) / v(1))`, and forecasts
`q_tau(h) = p1 * exp(quantile_tau of those log ratios)`, lightly smoothed
across neighbouring hours. Two guards keep the pool honest: the GTA VI
category's own pre-release history is not a launch and never enters the
pool, and launches whose hour 1 was still tiny next to their day peak
(late single-hour global unlocks, not rolling midnight-local console
launches) are excluded. Pool members are weighted by how closely their
pre-launch signature (the hottest pre-release hour relative to their
hour 1, read from the case's own history for the target) matches the
target's, and the median log ratio is shrunk 15% toward persistence —
both settings picked by leave-one-out cross-validation over the
comparables (CV log-pinball ≈ 0.29 on practice-like launches vs ≈ 0.37
for the persistence baseline there). Thin hours fall back to the
persistence spread. Deterministic, no network, runs in well under a
second per case.

## Run locally

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## License

MIT — see `LICENSE`.
