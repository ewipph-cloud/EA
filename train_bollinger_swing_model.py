import datetime as dt
import json
import math
from pathlib import Path

import train_swing_model as base


ROOT = Path(__file__).parent
DATA_PATH = ROOT / "XAUUSD_M5_HistData_2022-2026.csv"
MODEL_PATH = ROOT / "xauusd_m5_bollinger_swing_model.json"
SWING_SCALE = 1.5
LEG_COUNT = 3
CONTEXT_FEATURES = 6
BB_FEATURES = 3
MIN_COVERAGE = 0.10
MIN_SIGNALS = 100
BB_PERIOD = 20
BB_DEVIATIONS = 2.0


def calculate_bollinger_features(bars):
    closes = [bar[4] for bar in bars]
    widths = []
    percent_b = []
    running_sum = 0.0
    running_squares = 0.0
    slow_width = None

    for index, close in enumerate(closes):
        running_sum += close
        running_squares += close * close
        if index >= BB_PERIOD:
            old = closes[index - BB_PERIOD]
            running_sum -= old
            running_squares -= old * old
        if index + 1 < BB_PERIOD:
            widths.append(0.0)
            percent_b.append(0.5)
            continue

        mean = running_sum / BB_PERIOD
        variance = max(running_squares / BB_PERIOD - mean * mean, 0.0)
        deviation = math.sqrt(variance)
        width = 2.0 * BB_DEVIATIONS * deviation / max(abs(mean), 1e-8)
        position = (
            0.5
            if deviation <= 1e-12
            else (close - (mean - BB_DEVIATIONS * deviation))
            / (2.0 * BB_DEVIATIONS * deviation)
        )
        widths.append(width)
        percent_b.append(max(-1.0, min(2.0, position)))
        slow_width = width if slow_width is None else 0.02 * width + 0.98 * slow_width

    features = []
    slow_width = None
    for index, width in enumerate(widths):
        if index + 1 < BB_PERIOD:
            features.append((0.5, 0.0, 1.0))
            continue
        slow_width = width if slow_width is None else 0.02 * width + 0.98 * slow_width
        features.append(
            (
                percent_b[index],
                width,
                width / max(slow_width, 1e-8),
            )
        )
    return features


def add_context_and_bollinger(bars, scales, pivots, confirmation_index):
    features = original_pattern_features(
        bars, scales, pivots, confirmation_index
    )
    if features is None:
        return None

    source_time = dt.datetime.strptime(
        bars[confirmation_index][0], "%Y-%m-%d %H:%M:%S"
    ).replace(tzinfo=dt.timezone(dt.timedelta(hours=-5)))
    utc_time = source_time.astimezone(dt.timezone.utc)
    minute_of_day = utc_time.hour * 60 + utc_time.minute
    day_angle = 2.0 * math.pi * minute_of_day / 1440.0
    week_angle = 2.0 * math.pi * utc_time.weekday() / 7.0
    close = bars[confirmation_index][4]
    scale = max(scales[confirmation_index], 1e-8)
    one_hour_return = (
        (close - bars[confirmation_index - 12][4]) / scale
        if confirmation_index >= 12
        else 0.0
    )
    four_hour_return = (
        (close - bars[confirmation_index - 48][4]) / scale
        if confirmation_index >= 48
        else 0.0
    )
    features.extend(
        (
            math.sin(day_angle),
            math.cos(day_angle),
            math.sin(week_angle),
            math.cos(week_angle),
            one_hour_return,
            four_hour_return,
            *bollinger_features[confirmation_index],
        )
    )
    return features


def wilson_lower(successes, count):
    if not count:
        return 0.0
    z = 1.96
    proportion = successes / count
    denominator = 1.0 + z * z / count
    center = proportion + z * z / (2.0 * count)
    margin = z * math.sqrt(
        proportion * (1.0 - proportion) / count + z * z / (4.0 * count * count)
    )
    return (center - margin) / denominator


def selective_metrics(split, predictions, threshold, majority_class):
    selected = correct = true_up = true_down = false_up = false_down = 0
    for probability, actual in zip(predictions, split["y"]):
        prediction = int(probability >= 0.5)
        if max(probability, 1.0 - probability) < threshold:
            continue
        selected += 1
        correct += int(prediction == actual)
        if prediction == 1 and actual == 1:
            true_up += 1
        elif prediction == 0 and actual == 0:
            true_down += 1
        elif prediction == 1:
            false_up += 1
        else:
            false_down += 1

    up_count = true_up + false_down
    down_count = true_down + false_up
    return {
        "signals": selected,
        "coverage": selected / split["count"] if split["count"] else 0.0,
        "accuracy": correct / selected if selected else None,
        "wilson_95_lower": wilson_lower(correct, selected),
        "train_majority_baseline_accuracy": (
            up_count / selected if selected and majority_class else (
                down_count / selected if selected else None
            )
        ),
    }


def predict(split, weights, bias, means, scales):
    return [
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


def main():
    global original_pattern_features, bollinger_features
    base.DATA_PATH = DATA_PATH
    original_pattern_features = base.make_pattern_features
    bars = base.load_bars()
    bollinger_features = calculate_bollinger_features(bars)
    base.make_pattern_features = add_context_and_bollinger
    base.SWING_SCALE = SWING_SCALE
    base.LEG_COUNT = LEG_COUNT
    base.DIMENSIONS = (
        LEG_COUNT * base.FEATURES_PER_LEG + 2 + CONTEXT_FEATURES + BB_FEATURES
    )

    samples, _ = base.detect_samples(bars)
    rows, train_end, validation_end = base.split_samples(samples, len(bars))
    packed = {name: base.pack(split) for name, split in rows.items()}
    means, scales = base.standardize(packed["train"]["x"])
    weights, bias = base.train(packed["train"], means, scales)

    train_up_rate = sum(packed["train"]["y"]) / packed["train"]["count"]
    majority_class = int(train_up_rate >= 0.5)
    validation_predictions = predict(
        packed["validation"], weights, bias, means, scales
    )
    minimum = max(
        MIN_SIGNALS, math.ceil(packed["validation"]["count"] * MIN_COVERAGE)
    )
    thresholds = []
    for step in range(50, 100):
        threshold = step / 100.0
        metrics = selective_metrics(
            packed["validation"], validation_predictions, threshold, majority_class
        )
        if metrics["signals"] >= minimum:
            thresholds.append((metrics["accuracy"], metrics["coverage"], threshold, metrics))
    if not thresholds:
        raise RuntimeError("No validation threshold meets the minimum signal count")

    _, _, threshold, validation_metrics = max(thresholds)
    test_predictions = predict(packed["test"], weights, bias, means, scales)
    test_metrics = selective_metrics(
        packed["test"], test_predictions, threshold, majority_class
    )
    model = {
        "model": "logistic regression on variable swing patterns, market context, and Bollinger Bands",
        "source": "HistData XAU/USD M1 resampled to M5",
        "pattern": {
            "swing_scale": SWING_SCALE,
            "leg_count": LEG_COUNT,
            "duration": "variable; defined by volatility-scaled directional changes",
        },
        "bollinger_bands": {
            "period": BB_PERIOD,
            "standard_deviations": BB_DEVIATIONS,
            "features": [
                "percent_b",
                "bandwidth_over_middle",
                "bandwidth_over_ewma_bandwidth",
            ],
            "calculation": "Uses only the current and preceding closed candles.",
        },
        "other_context": [
            "UTC time-of-day and day-of-week cycles",
            "prior 1-hour and 4-hour price returns normalized by local range",
        ],
        "outcome": {
            "method": "first touch of either volatility-scaled price barrier",
            "barrier_scale": base.OUTCOME_SCALE,
            "max_observation_bars": base.MAX_OUTCOME_BARS,
        },
        "split": {
            "train_end_index": train_end,
            "validation_end_index": validation_end,
            "train_samples": packed["train"]["count"],
            "validation_samples": packed["validation"]["count"],
            "test_samples": packed["test"]["count"],
        },
        "validation_threshold": threshold,
        "minimum_validation_signals": minimum,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "warning": "Test interval previously inspected in earlier experiments; diagnostic only.",
        "normalization_means": means,
        "normalization_scales": scales,
        "weights": weights,
        "bias": bias,
    }
    MODEL_PATH.write_text(json.dumps(model, indent=2))
    print(
        json.dumps(
            {
                "model_path": str(MODEL_PATH),
                "threshold": threshold,
                "train_samples": packed["train"]["count"],
                "validation_metrics": validation_metrics,
                "test_metrics": test_metrics,
                "warning": model["warning"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
