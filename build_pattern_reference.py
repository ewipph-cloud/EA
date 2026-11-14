import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import train_swing_model as base


ROOT = Path(__file__).parent
DATA_PATH = ROOT / "XAUUSD_M5_HistData_2022-2026.csv"
REFERENCE_CSV = ROOT / "xauusd_pattern_reference.csv"
SUMMARY_JSON = ROOT / "xauusd_pattern_reference_summary.json"
TRAIN_CUTOFF = "2026-01-01 00:00:00"
SWING_SCALE = 2.0
LEG_COUNT = 3
MIN_TRAIN_SUPPORT = 50
MIN_TEST_SUPPORT = 20
STRONG_TRAIN_RATE = 0.60


def candle_shape(bar):
    _, open_price, high, low, close = bar
    span = max(high - low, 1e-8)
    body = abs(close - open_price) / span
    upper = (high - max(open_price, close)) / span
    lower = (min(open_price, close) - low) / span
    if body <= 0.20:
        return "small_body"
    if lower >= 0.50 and lower >= 1.5 * upper:
        return "lower_rejection"
    if upper >= 0.50 and upper >= 1.5 * lower:
        return "upper_rejection"
    return "bull_body" if close > open_price else "bear_body"


def pattern_key(bars, scales, pivots, confirmation_index):
    recent = pivots[-(LEG_COUNT + 1) :]
    if len(recent) != LEG_COUNT + 1:
        return None

    amplitudes = []
    directions = []
    for start, end in zip(recent, recent[1:]):
        start_index, start_price = start
        end_index, end_price = end
        if end_index <= start_index:
            return None
        directions.append("U" if end_price > start_price else "D")
        amplitudes.append(
            abs(end_price - start_price) / max(scales[end_index], 1e-8)
        )

    ratio = amplitudes[-1] / max(amplitudes[0], 1e-8)
    if ratio < 0.67:
        size_shape = "contracting"
    elif ratio > 1.50:
        size_shape = "expanding"
    else:
        size_shape = "mixed"

    shape = candle_shape(bars[confirmation_index])
    return f'{"-".join(directions)}|{size_shape}|{shape}'


def first_barrier_outcome(bars, scales, start_index):
    start_price = bars[start_index][4]
    barrier = max(scales[start_index] * base.OUTCOME_SCALE, 1e-8)
    stop = min(start_index + base.MAX_OUTCOME_BARS, len(bars) - 1)
    for index in range(start_index + 1, stop + 1):
        hits_up = bars[index][2] >= start_price + barrier
        hits_down = bars[index][3] <= start_price - barrier
        if hits_up and hits_down:
            return None
        if hits_up:
            return 1, index
        if hits_down:
            return 0, index
    return None


def collect_events(bars):
    scales = base.calculate_scales(bars)
    events = []
    pivots = []
    direction = 0
    initial_index = 0
    initial_price = bars[0][4]
    extreme_index = 0
    extreme_price = initial_price

    for index in range(1, len(bars)):
        threshold = max(scales[index - 1] * SWING_SCALE, 1e-8)
        close = bars[index][4]
        confirmed_pivot = None

        if direction == 0:
            if close - initial_price >= threshold:
                direction = 1
                extreme_index, extreme_price = index, close
            elif initial_price - close >= threshold:
                direction = -1
                extreme_index, extreme_price = index, close
            continue

        if direction > 0:
            if close > extreme_price:
                extreme_index, extreme_price = index, close
            if extreme_price - close >= threshold:
                confirmed_pivot = (extreme_index, extreme_price)
                direction = -1
                extreme_index, extreme_price = index, close
        else:
            if close < extreme_price:
                extreme_index, extreme_price = index, close
            if close - extreme_price >= threshold:
                confirmed_pivot = (extreme_index, extreme_price)
                direction = 1
                extreme_index, extreme_price = index, close

        if confirmed_pivot is None:
            continue
        if not pivots:
            pivots.append((initial_index, initial_price))
        pivots.append(confirmed_pivot)
        key = pattern_key(bars, scales, pivots, index)
        outcome = first_barrier_outcome(bars, scales, index)
        if key is not None and outcome is not None:
            label, resolved_index = outcome
            events.append(
                {
                    "time": bars[index][0],
                    "index": index,
                    "resolved_index": resolved_index,
                    "resolved_time": bars[resolved_index][0],
                    "pattern": key,
                    "outcome_up": label,
                }
            )
    return events


def wilson_interval(successes, count):
    if not count:
        return None, None
    z = 1.96
    p = successes / count
    denominator = 1.0 + z * z / count
    center = p + z * z / (2.0 * count)
    margin = z * math.sqrt(p * (1.0 - p) / count + z * z / (4.0 * count * count))
    return (center - margin) / denominator, (center + margin) / denominator


def summarize(events):
    train = defaultdict(lambda: [0, 0])
    test = defaultdict(lambda: [0, 0])
    for event in events:
        if event["time"] < TRAIN_CUTOFF:
            if event["resolved_time"] >= TRAIN_CUTOFF:
                continue
            group = train
        else:
            group = test
        group[event["pattern"]][0] += 1
        group[event["pattern"]][1] += event["outcome_up"]

    library = []
    for pattern, (train_count, train_up) in train.items():
        test_count, test_up = test.get(pattern, (0, 0))
        if train_count < MIN_TRAIN_SUPPORT or test_count < MIN_TEST_SUPPORT:
            continue
        train_rate = train_up / train_count
        test_rate = test_up / test_count
        lower, upper = wilson_interval(test_up, test_count)
        library.append(
            {
                "pattern": pattern,
                "train_events": train_count,
                "train_up_rate": train_rate,
                "test_events": test_count,
                "test_up_rate": test_rate,
                "test_wilson_95_low": lower,
                "test_wilson_95_high": upper,
                "train_majority_prediction": "up" if train_rate >= 0.5 else "down",
                "test_train_rate_difference": test_rate - train_rate,
            }
        )
    library.sort(key=lambda row: row["test_events"], reverse=True)

    eligible = [
        row
        for row in library
        if row["train_up_rate"] >= STRONG_TRAIN_RATE
        or row["train_up_rate"] <= 1.0 - STRONG_TRAIN_RATE
    ]
    eligible_by_pattern = {row["pattern"]: row for row in eligible}
    eligible_test_events = [
        event
        for event in events
        if event["time"] >= TRAIN_CUTOFF
        and event["pattern"] in eligible_by_pattern
    ]
    correct = 0
    for event in eligible_test_events:
        row = eligible_by_pattern[event["pattern"]]
        prediction = int(row["train_majority_prediction"] == "up")
        correct += int(prediction == event["outcome_up"])

    test_events = [event for event in events if event["time"] >= TRAIN_CUTOFF]
    return library, {
        "total_resolved_events": len(events),
        "train_events_before_2026": sum(v[0] for v in train.values()),
        "test_events_2026": len(test_events),
        "patterns_meeting_support": len(library),
        "strong_reference_patterns": len(eligible),
        "strong_pattern_test_coverage": (
            len(eligible_test_events) / len(test_events) if test_events else 0.0
        ),
        "strong_pattern_test_accuracy": (
            correct / len(eligible_test_events) if eligible_test_events else None
        ),
        "split": {
            "reference_period": f"before {TRAIN_CUTOFF}",
            "evaluation_period": f"from {TRAIN_CUTOFF}",
        },
        "pattern_definition": {
            "variable_duration_swings": LEG_COUNT,
            "swing_reversal_scale": SWING_SCALE,
            "size_shape_categories": ["contracting", "mixed", "expanding"],
            "candle_shape_categories": [
                "small_body",
                "lower_rejection",
                "upper_rejection",
                "bull_body",
                "bear_body",
            ],
        },
        "outcome_definition": {
            "first_touch_barrier_scale": base.OUTCOME_SCALE,
            "maximum_observation_bars": base.MAX_OUTCOME_BARS,
            "source_note": "Bid-only price; excludes spread, slippage, and volume.",
        },
    }


def main():
    base.DATA_PATH = DATA_PATH
    bars = base.load_bars()
    events = collect_events(bars)
    library, summary = summarize(events)
    with REFERENCE_CSV.open("w", newline="") as target:
        fields = [
            "pattern",
            "train_events",
            "train_up_rate",
            "test_events",
            "test_up_rate",
            "test_wilson_95_low",
            "test_wilson_95_high",
            "train_majority_prediction",
            "test_train_rate_difference",
        ]
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows(library)
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2))

    top = sorted(library, key=lambda row: row["test_events"], reverse=True)[:12]
    print(
        json.dumps(
            {
                "reference_csv": str(REFERENCE_CSV),
                "summary_json": str(SUMMARY_JSON),
                "summary": summary,
                "top_recurrent_patterns": top,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
