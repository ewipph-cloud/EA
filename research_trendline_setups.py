import datetime as dt
import json
import math
from collections import defaultdict
from pathlib import Path

import train_swing_model as base


ROOT = Path(__file__).parent
DATA_PATH = ROOT / "XAUUSD_M5_HistData_2022-2026.csv"
OUTPUT_PATH = ROOT / "xauusd_trendline_setups_research.json"
SWING_SCALE = 1.5
MIN_SIGNALS_PER_GROUP = 30


def line_from_points(points, confirmation_index):
    if len(points) < 2:
        return None
    mean_x = sum(index for index, _ in points) / len(points)
    mean_y = sum(price for _, price in points) / len(points)
    variance = sum((index - mean_x) ** 2 for index, _ in points)
    if variance <= 1e-12:
        return None
    slope = sum(
        (index - mean_x) * (price - mean_y) for index, price in points
    ) / variance
    intercept = mean_y - slope * mean_x
    error = math.sqrt(
        sum(
            (price - (intercept + slope * index)) ** 2
            for index, price in points
        )
        / len(points)
    )
    return {
        "level": intercept + slope * confirmation_index,
        "previous_level": intercept + slope * (confirmation_index - 1),
        "slope": slope,
        "error": error,
        "points": len(points),
    }


def recent_trendlines(bars, pivots, confirmation_index, scale):
    highs = []
    lows = []
    first_position = max(1, len(pivots) - 6)
    for position in range(first_position, len(pivots)):
        start_index, start_price = pivots[position - 1]
        end_index, end_price = pivots[position]
        if end_index <= start_index:
            continue

        if end_price < start_price:
            segment = range(start_index, end_index + 1)
            high_index = max(segment, key=lambda index: bars[index][2])
            highs.append((high_index, bars[high_index][2]))
        else:
            segment = range(start_index, end_index + 1)
            low_index = min(segment, key=lambda index: bars[index][3])
            lows.append((low_index, bars[low_index][3]))

    scale = max(scale, 1e-8)
    resistance = line_from_points(highs[-3:], confirmation_index)
    support = line_from_points(lows[-3:], confirmation_index)
    close = bars[confirmation_index][4]
    candle_high = bars[confirmation_index][2]
    candle_low = bars[confirmation_index][3]
    result = {}
    for name, line in (("support", support), ("resistance", resistance)):
        if line is None:
            result[name] = None
            continue
        line["normalized_slope_per_hour"] = line["slope"] * 12.0 / scale
        line["normalized_error"] = line["error"] / scale
        line["close_distance"] = (close - line["level"]) / scale
        line["nearby"] = (
            candle_low <= line["level"] + 0.5 * scale
            and candle_high >= line["level"] - 0.5 * scale
        )
        result[name] = line

    if support:
        result["breakdown"] = (
            bars[confirmation_index - 1][4]
            >= support["previous_level"] - 0.1 * scale
            and close < support["level"] - 0.2 * scale
        )
    else:
        result["breakdown"] = False
    if resistance:
        result["breakout"] = (
            bars[confirmation_index - 1][4]
            <= resistance["previous_level"] + 0.1 * scale
            and close > resistance["level"] + 0.2 * scale
        )
    else:
        result["breakout"] = False
    result["scale"] = scale
    return result


def setup_rules(lines):
    if not lines:
        return {}
    support = lines["support"]
    resistance = lines["resistance"]
    rules = {
        "support_rejection_uptrend": False,
        "support_rejection_any_trend": False,
        "resistance_rejection_downtrend": False,
        "resistance_rejection_any_trend": False,
        "confirmed_resistance_breakout": False,
        "confirmed_support_breakdown": False,
        "any_support_or_resistance_touch": False,
    }
    if support:
        near = (
            support["nearby"]
            and support["normalized_error"] <= 1.0
            and support["close_distance"] >= 0.0
        )
        rules["support_rejection_any_trend"] = near
        rules["support_rejection_uptrend"] = (
            near and support["normalized_slope_per_hour"] >= 0.05
        )
        rules["any_support_or_resistance_touch"] |= support["nearby"]
    if resistance:
        near = (
            resistance["nearby"]
            and resistance["normalized_error"] <= 1.0
            and resistance["close_distance"] <= 0.0
        )
        rules["resistance_rejection_any_trend"] = near
        rules["resistance_rejection_downtrend"] = (
            near and resistance["normalized_slope_per_hour"] <= -0.05
        )
        rules["any_support_or_resistance_touch"] |= resistance["nearby"]
    rules["confirmed_resistance_breakout"] = bool(
        lines["breakout"]
        and resistance
        and resistance["normalized_error"] <= 1.0
    )
    rules["confirmed_support_breakdown"] = bool(
        lines["breakdown"]
        and support
        and support["normalized_error"] <= 1.0
    )
    return rules


def collect_setups(bars, scales, pivots, confirmation_index):
    lines = recent_trendlines(
        bars, pivots, confirmation_index, scales[confirmation_index]
    )
    events[confirmation_index] = {
        "rules": setup_rules(lines),
        "lines": lines,
    }
    return [0.0]


def base_wilson_lower(successes, count):
    if not count:
        return 0.0
    z = 1.96
    proportion = successes / count
    denominator = 1.0 + z * z / count
    center = proportion + z * z / (2.0 * count)
    margin = z * math.sqrt(
        proportion * (1.0 - proportion) / count
        + z * z / (4.0 * count * count)
    )
    return (center - margin) / denominator


def period_name(timestamp):
    parsed = dt.datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
    quarter = (parsed.month - 1) // 3 + 1
    return f"{parsed.year}-Q{quarter}"


def main():
    global events
    base.DATA_PATH = DATA_PATH
    base.SWING_SCALE = SWING_SCALE
    bars = base.load_bars()
    scales = base.calculate_scales(bars)
    events = {}
    base.make_pattern_features = collect_setups
    samples, _ = base.detect_samples(bars)
    split_rows, train_end, validation_end = base.split_samples(
        samples, len(bars)
    )
    split_by_event = {
        sample["event_index"]: name
        for name, rows in split_rows.items()
        for sample in rows
    }
    split_baselines = {}
    for name, rows in split_rows.items():
        if not rows:
            split_baselines[name] = None
            continue
        up_rate = sum(sample["label"] for sample in rows) / len(rows)
        split_baselines[name] = max(up_rate, 1.0 - up_rate)
    all_up_rate = sum(sample["label"] for sample in samples) / len(samples)
    all_events_baseline = max(all_up_rate, 1.0 - all_up_rate)
    quarter_outcomes = defaultdict(lambda: {"count": 0, "up": 0})
    for sample in samples:
        quarter = period_name(bars[sample["event_index"]][0])
        quarter_outcomes[quarter]["count"] += 1
        quarter_outcomes[quarter]["up"] += sample["label"]

    records = []
    for sample in samples:
        data = events.get(sample["event_index"])
        if data is None:
            continue
        for rule_name, active in data["rules"].items():
            if not active:
                continue
            if rule_name in (
                "support_rejection_uptrend",
                "support_rejection_any_trend",
                "confirmed_resistance_breakout",
            ):
                direction = 1
            elif rule_name in (
                "resistance_rejection_downtrend",
                "resistance_rejection_any_trend",
                "confirmed_support_breakdown",
            ):
                direction = 0
            else:
                continue
            records.append(
                {
                    "event_index": sample["event_index"],
                    "split": split_by_event.get(sample["event_index"], "excluded"),
                    "period": period_name(bars[sample["event_index"]][0]),
                    "rule": rule_name,
                    "direction": direction,
                    "label": sample["label"],
                }
            )

    rules = sorted({row["rule"] for row in records})
    report = {}
    for rule in rules:
        selected = [row for row in records if row["rule"] == rule]
        def summarize_group(group, overall_baseline):
            total = len(group)
            if not total:
                return {"signals": 0, "accuracy": None}
            correct = sum(row["label"] == row["direction"] for row in group)
            up_rate = sum(row["label"] == 1 for row in group) / total
            return {
                "signals": total,
                "accuracy": correct / total,
                "wilson_95_lower": base_wilson_lower(correct, total),
                "outcome_up_rate": up_rate,
                "majority_direction_baseline_for_this_setup": max(
                    up_rate, 1.0 - up_rate
                ),
                "all_events_majority_baseline": overall_baseline,
                "edge_vs_all_events_baseline": (
                    correct / total - overall_baseline
                    if overall_baseline is not None
                    else None
                ),
            }

        # A quarter-level report makes temporal instability visible; low-count
        # quarters are retained with their counts but not interpreted as rates.
        by_quarter = defaultdict(list)
        for row in selected:
            by_quarter[row["period"]].append(row)
        quarter_stats = {}
        for quarter, quarter_rows in sorted(by_quarter.items()):
            total = len(quarter_rows)
            correct = sum(row["label"] == row["direction"] for row in quarter_rows)
            quarter_up_rate = sum(row["label"] for row in quarter_rows) / total
            quarter_baseline = max(quarter_up_rate, 1.0 - quarter_up_rate)
            quarter_outcomes_row = quarter_outcomes[quarter]
            all_up_rate = quarter_outcomes_row["up"] / quarter_outcomes_row["count"]
            quarter_all_events_baseline = max(all_up_rate, 1.0 - all_up_rate)
            quarter_stats[quarter] = {
                "signals": total,
                "accuracy": correct / total if total else None,
                "all_events_majority_baseline": quarter_all_events_baseline,
                "edge_vs_all_events_baseline": (
                    correct / total - quarter_all_events_baseline
                    if total
                    else None
                ),
                "reliable_sample": total >= MIN_SIGNALS_PER_GROUP,
            }
        report[rule] = {
            "overall": summarize_group(selected, all_events_baseline),
            "chronological_splits": {
                name: summarize_group(
                    [row for row in selected if row["split"] == name],
                    split_baselines.get(name),
                )
                for name in ("train", "validation", "test")
            },
            "quarters": quarter_stats,
        }

    payload = {
        "dataset": str(DATA_PATH),
        "bars": len(bars),
        "resolved_events": len(samples),
        "chronological_split_indices": {
            "train_end_index": train_end,
            "validation_end_index": validation_end,
        },
        "outcome_definition": (
            "Direction of first touch of symmetric volatility-scaled barrier; "
            "not a net-profit measure and excludes transaction costs."
        ),
        "trendline_method": (
            "Up to three confirmed swing highs/lows from completed legs; "
            "touch and breakout rules use current/past OHLC only."
        ),
        "rules": report,
        "warning": (
            "Quarterly groups overlap in clustered event outcomes and are not "
            "independent samples. Use as exploratory evidence, not proof or "
            "a live trading rule."
        ),
    }
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2))
    print(json.dumps({"output_path": str(OUTPUT_PATH), **payload}, indent=2))


if __name__ == "__main__":
    main()
