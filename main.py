"""Launch-day Twitch CCU forecaster for gta6-twitch-ccu-forecast-01.

Model: empirical ratio curves from the case's comparable launches.

For every comparable launch we rebuild its release-day curve (UTC day that
starts at the comparable's ``day_start``) and record, for each hour h, the
log ratio log(v(h) / v(1)).  The forecast for a target hour is the target's
own hour-1 print times exp() of the matching empirical quantile of those
log ratios, so the whole intraday shape -- the morning climb as later
time zones unlock, the prime-time peak, the late-night fade -- comes from
what past big launches actually did, hour by hour.

Two guards keep the pool honest:

* the Grand Theft Auto VI category's own pre-release history rides along
  in ``comparables`` but is not a launch (its unlock note says so, and its
  "day" is a trailer premiere), so it never enters the pool;
* launches whose hour 1 was still tiny next to their own day peak
  (v(1) / peak < 0.10) unlocked late in the UTC day -- a single global
  unlock hour, not the rolling midnight-local pattern of the console
  launches this contest is about.  Their ratios explode by orders of
  magnitude and would blow up the upper quantiles, so they are left out.

The median log ratio is shrunk 15% toward zero (plain persistence): in
leave-one-out cross-validation over the comparables that slightly beat
using the raw median, because a target's own peak tends to sit a little
closer to its first hour than the pooled median launch's does.

Hours whose pool is too thin fall back to the persistence spread
q = p1 * exp(z * 0.15 * sqrt(h - 1)).  Everything is deterministic:
sorted inputs, no randomness, no clock, no network, stdlib only.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]
FALLBACK_SIGMA = 0.15
MIN_START_RATIO = 0.10
MEDIAN_SHRINK = 0.85
MIN_POOL = 8
MIN_DAY_HOURS = 20


def round2(x: float) -> float:
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    return whole / 100


def parse_ts(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc)


def day_curve(comp: dict) -> dict:
    """Hour-of-release-day (1..24) -> viewers for one comparable."""
    start = parse_ts(comp["day_start"])
    curve = {}
    for point in comp.get("hourly", []):
        viewers = point.get("viewers")
        if not viewers or viewers <= 0:
            continue
        hour = int((parse_ts(point["from"]) - start).total_seconds() // 3600) + 1
        if 1 <= hour <= 24:
            curve[hour] = float(viewers)
    return curve


def ratio_pool(case: dict) -> dict:
    """Hour -> sorted list of log ratios log(v(h)/v(1)) over the pool."""
    pool: dict[int, list[float]] = {}
    for comp in case.get("comparables", []):
        if str(comp.get("unlock", "")).startswith("Not a launch"):
            continue
        curve = day_curve(comp)
        if 1 not in curve or len(curve) < MIN_DAY_HOURS:
            continue
        first = curve[1]
        if first / max(curve.values()) < MIN_START_RATIO:
            continue
        for hour in range(2, 25):
            if hour in curve:
                pool.setdefault(hour, []).append(math.log(curve[hour] / first))
    for values in pool.values():
        values.sort()
    return pool


def quantile(sorted_values: list[float], tau: float) -> float:
    n = len(sorted_values)
    if n == 1:
        return sorted_values[0]
    pos = tau * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def smoothed_log_quantiles(pool: dict) -> dict:
    """Hour -> the five log-ratio quantiles, lightly smoothed over time.

    Each hour's quantiles are averaged with its neighbours' (in log space)
    before use.  Adjacent hours of a launch day move together, so this
    trims the sampling noise of a ~27-launch pool without flattening the
    day's real shape; cross-validation found it cost nothing and helped
    a hair.  Hours with a thin pool are absent here and fall back later.
    """
    raw = {}
    for hour in range(2, 25):
        values = pool.get(hour, [])
        if len(values) >= MIN_POOL:
            raw[hour] = [quantile(values, tau) for tau in LEVELS]
    smoothed = {}
    for hour, logs in raw.items():
        neighbours = [raw[h] for h in (hour - 1, hour, hour + 1) if h in raw]
        smoothed[hour] = [
            sum(n[k] for n in neighbours) / len(neighbours) for k in range(len(LEVELS))
        ]
    return smoothed


def hour_quantiles(p1: float, hour: int, log_quantiles: dict) -> list[float]:
    logs = log_quantiles.get(hour)
    if logs is not None:
        median = logs[2]
        raw = [p1 * math.exp(MEDIAN_SHRINK * median + (lq - median)) for lq in logs]
    else:
        raw = [p1 * math.exp(z * FALLBACK_SIGMA * math.sqrt(hour - 1)) for z in Z]
    out = []
    prev = 0.0
    for value in raw:
        value = min(max(round2(value), 0.01), 1_000_000_000)
        value = max(value, prev)
        out.append(value)
        prev = value
    return out


def forecast_case(case: dict) -> dict:
    first = next(k for k in case["known"] if k["hour"] == 1)
    p1 = max(1.0, float(first["viewers"]))
    log_quantiles = smoothed_log_quantiles(ratio_pool(case))
    rows = [
        {"hour": row["hour"], "q": hour_quantiles(p1, row["hour"], log_quantiles)}
        for row in case["forecast"]
    ]
    rows.sort(key=lambda r: r["hour"])
    return {"hours": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        result = forecast_case(case)
        payload = json.dumps(result, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / f"{path.stem}.out").write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()
