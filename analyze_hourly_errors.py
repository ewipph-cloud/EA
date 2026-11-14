import datetime as dt
import json
from pathlib import Path

import research_indicator_combination as research
import train_swing_model as base


ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "xauusd_m5_combined_indicators_model.json"
OUTPUT_PATH = ROOT / "xauusd_hourly_error_analysis.json"
SWING_SCALE = research.SWING_SCALE
LEG_COUNT = research.LEG_COUNT


def add_all_features(bars, scales, pivots, confirmation_index):
    features = research.add_base_context(
        bars, scales, pivots, confirmation_index
    )
    if features is None:
        return None
    features.extend(research.market_features[confirmation_index]["candle"])
    features.extend(research.market_features[confirmation_index]["indicators"])
    features.extend(
        research.market_features[confirmation_index]["interactions"]
    )
    return features


def hourly_metrics(samples, probabilities, threshold):
    hours = {
        hour: {"events": 0, "signals": 0, "correct": 0}
        for hour in range(24)
    }
    for sample, probability in zip(samples, probabilities):
        time = dt.datetime.strptime(
            base_time[sample["event_index"]], "%Y-%m-%d %H:%M:%S"
        )
        row = hours[time.hour]
        row["events"] += 1
        if max(probability, 1.0 - probability) < threshold:
            continue
        row["signals"] += 1
        row["correct"] += int((probability >= 0.5) == sample["label"])

    return {
        f"{hour:02d}:00": {
            **row,
            "signal_coverage": row["signals"] / row["events"] if row["events"] else 0,
            "accuracy": row["correct"] / row["signals"] if row["signals"] else None,
        }
        for hour, row in hours.items()
    }


def main():
    global base_time
    model = json.loads(MODEL_PATH.read_text())
    base.DATA_PATH = research.DATA_PATH
    base.SWING_SCALE = SWING_SCALE
    base.LEG_COUNT = LEG_COUNT

    bars = base.load_bars()
    base_time = [bar[0] for bar in bars]
    range_scales = base.calculate_scales(bars)
    research.original_pattern_features = base.make_pattern_features
    research.market_features = research.calculate_market_features(
        bars, range_scales
    )
    base.make_pattern_features = add_all_features
    base.DIMENSIONS = (
        LEG_COUNT * base.FEATURES_PER_LEG + 2 + research.CONTEXT_FEATURES + 4 + 9 + 8
    )

    samples, _ = base.detect_samples(bars)
    split_rows, _, _ = base.split_samples(samples, len(bars))
    packed = {name: base.pack(rows) for name, rows in split_rows.items()}
    means = model["normalization_means"]
    scales = model["normalization_scales"]
    weights = model["weights"]
    bias = model["bias"]
    threshold = model["validation_threshold"]
    report = {}
    for name in ("validation", "test"):
        split = packed[name]
        probabilities = [
            base.probability(
                weights,
                bias,
                split["x"],
                sample * base.DIMENSIONS,
                means,
                scales,
            )
            for sample in range(split["count"])
        ]
        report[name] = hourly_metrics(
            split_rows[name], probabilities, threshold
        )

    payload = {
        "threshold": threshold,
        "timezone": "dataset time_est (fixed EST; not DST-adjusted)",
        "validation": report["validation"],
        "test_diagnostic_only": report["test"],
        "warning": (
            "Hourly slices have small samples and multiple comparisons; "
            "the test interval was previously inspected."
        ),
    }
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2))
    print(json.dumps({"output_path": str(OUTPUT_PATH), **payload}, indent=2))


if __name__ == "__main__":
    main()
