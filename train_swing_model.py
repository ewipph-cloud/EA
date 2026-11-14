import array
import csv
import json
import math
import random
from pathlib import Path


ROOT = Path(__file__).parent
DATA_PATH = ROOT / "XAUUSD_M5_HistData_2025-2026.csv"
MODEL_PATH = ROOT / "xauusd_m5_swing_model.json"
LEG_COUNT = 4
SHAPE_POINTS = 6
FEATURES_PER_LEG = 4 + SHAPE_POINTS
DIMENSIONS = LEG_COUNT * FEATURES_PER_LEG + 2
EWMA_ALPHA = 0.04
SWING_SCALE = 2.0
OUTCOME_SCALE = 2.0
MAX_OUTCOME_BARS = 144
BATCH_SIZE = 256
EPOCHS = 10
LEARNING_RATE = 0.05
L2 = 0.002


def load_bars():
    bars = []
    with DATA_PATH.open(newline="") as source:
        for row in csv.DictReader(source):
            bars.append(
                (
                    row["time_est"],
                    float(row["open"]),
                    float(row["high"]),
                    float(row["low"]),
                    float(row["close"]),
                )
            )
    if len(bars) <= MAX_OUTCOME_BARS:
        raise RuntimeError(f"Not enough candle data in {DATA_PATH}")
    return bars


def calculate_scales(bars):
    value = max(bars[0][2] - bars[0][3], 1e-8)
    scales = []
    for bar in bars:
        value = EWMA_ALPHA * max(bar[2] - bar[3], 1e-8) + (1.0 - EWMA_ALPHA) * value
        scales.append(value)
    return scales


def describe_leg(bars, scales, start, end):
    start_index, start_price = start
    end_index, end_price = end
    if end_index <= start_index:
        return [0.0] * FEATURES_PER_LEG

    scale = max(scales[end_index], 1e-8)
    direction = 1.0 if end_price > start_price else -1.0
    amplitude = abs(end_price - start_price) / scale
    duration = math.log1p((end_index - start_index) * 5.0)
    path_length = sum(
        abs(bars[i][4] - bars[i - 1][4])
        for i in range(start_index + 1, end_index + 1)
    )
    efficiency = abs(end_price - start_price) / max(path_length, 1e-8)

    shape = []
    for point in range(1, SHAPE_POINTS + 1):
        fraction = point / (SHAPE_POINTS + 1)
        index = round(start_index + fraction * (end_index - start_index))
        index = min(max(index, start_index), end_index)
        shape.append((bars[index][4] - start_price) / scale)
    return [direction, amplitude, duration, efficiency, *shape]


def make_pattern_features(bars, scales, pivots, confirmation_index):
    recent = pivots[-(LEG_COUNT + 1) :]
    if len(recent) != LEG_COUNT + 1:
        return None

    features = []
    for start, end in zip(recent, recent[1:]):
        features.extend(describe_leg(bars, scales, start, end))

    latest_pivot_index, latest_pivot_price = recent[-1]
    scale = max(scales[confirmation_index], 1e-8)
    confirmation_move = (
        bars[confirmation_index][4] - latest_pivot_price
    ) / scale
    confirmation_duration = math.log1p(
        max(confirmation_index - latest_pivot_index, 0) * 5.0
    )
    features.extend((confirmation_move, confirmation_duration))
    return features


def first_barrier_outcome(bars, scales, start_index):
    start_price = bars[start_index][4]
    barrier = max(scales[start_index] * OUTCOME_SCALE, 1e-8)
    stop = min(start_index + MAX_OUTCOME_BARS, len(bars) - 1)
    for index in range(start_index + 1, stop + 1):
        up = bars[index][2] >= start_price + barrier
        down = bars[index][3] <= start_price - barrier
        if up and down:
            return None
        if up:
            return 1, index
        if down:
            return 0, index
    return None


def detect_samples(bars):
    scales = calculate_scales(bars)
    samples = []
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

        features = make_pattern_features(bars, scales, pivots, index)
        outcome = first_barrier_outcome(bars, scales, index)
        if features is not None and outcome is not None:
            label, resolved_index = outcome
            samples.append(
                {
                    "event_index": index,
                    "resolved_index": resolved_index,
                    "features": features,
                    "label": label,
                }
            )

    return samples, scales


def split_samples(samples, bar_count):
    train_end = int(bar_count * 0.70)
    validation_end = int(bar_count * 0.85)
    splits = {name: [] for name in ("train", "validation", "test")}
    for sample in samples:
        event_index = sample["event_index"]
        resolved_index = sample["resolved_index"]
        if resolved_index < train_end:
            splits["train"].append(sample)
        elif event_index >= train_end and resolved_index < validation_end:
            splits["validation"].append(sample)
        elif event_index >= validation_end:
            splits["test"].append(sample)
    return splits, train_end, validation_end


def pack(samples):
    x = array.array("d")
    y = array.array("b")
    for sample in samples:
        x.extend(sample["features"])
        y.append(sample["label"])
    return {"x": x, "y": y, "count": len(y)}


def standardize(train_x):
    count = len(train_x) // DIMENSIONS
    means = [0.0] * DIMENSIONS
    for sample in range(count):
        offset = sample * DIMENSIONS
        for feature in range(DIMENSIONS):
            means[feature] += train_x[offset + feature]
    means = [value / count for value in means]

    variances = [0.0] * DIMENSIONS
    for sample in range(count):
        offset = sample * DIMENSIONS
        for feature in range(DIMENSIONS):
            difference = train_x[offset + feature] - means[feature]
            variances[feature] += difference * difference
    scales = [math.sqrt(value / count) or 1.0 for value in variances]
    return means, scales


def probability(weights, bias, x, offset, means, scales):
    score = bias
    for feature in range(DIMENSIONS):
        normalized = (x[offset + feature] - means[feature]) / scales[feature]
        score += weights[feature] * normalized
    score = max(-35.0, min(35.0, score))
    return 1.0 / (1.0 + math.exp(-score))


def train(train_split, means, scales):
    x = train_split["x"]
    y = train_split["y"]
    count = train_split["count"]
    weights = [0.0] * DIMENSIONS
    bias = 0.0
    order = list(range(count))
    rng = random.Random(20261112)

    for _ in range(EPOCHS):
        rng.shuffle(order)
        for batch_start in range(0, count, BATCH_SIZE):
            batch = order[batch_start : batch_start + BATCH_SIZE]
            gradient = [0.0] * DIMENSIONS
            bias_gradient = 0.0
            for sample in batch:
                offset = sample * DIMENSIONS
                score = bias
                normalized_features = []
                for feature in range(DIMENSIONS):
                    normalized = (x[offset + feature] - means[feature]) / scales[feature]
                    normalized_features.append(normalized)
                    score += weights[feature] * normalized
                score = max(-35.0, min(35.0, score))
                error = 1.0 / (1.0 + math.exp(-score)) - y[sample]
                bias_gradient += error
                for feature, normalized in enumerate(normalized_features):
                    gradient[feature] += error * normalized

            rate = LEARNING_RATE / len(batch)
            bias -= rate * bias_gradient
            for feature in range(DIMENSIONS):
                weights[feature] -= rate * (
                    gradient[feature] + L2 * weights[feature] * len(batch)
                )
    return weights, bias


def evaluate(split, weights, bias, means, scales, majority_class):
    y = split["y"]
    x = split["x"]
    count = split["count"]
    true_up = true_down = false_up = false_down = 0
    confident_count = confident_correct = 0

    for sample in range(count):
        p_up = probability(
            weights, bias, x, sample * DIMENSIONS, means, scales
        )
        prediction = int(p_up >= 0.5)
        actual = y[sample]
        if prediction == 1 and actual == 1:
            true_up += 1
        elif prediction == 0 and actual == 0:
            true_down += 1
        elif prediction == 1:
            false_up += 1
        else:
            false_down += 1
        if max(p_up, 1.0 - p_up) >= 0.55:
            confident_count += 1
            confident_correct += int(prediction == actual)

    up_count = true_up + false_down
    down_count = true_down + false_up
    accuracy = (true_up + true_down) / count
    recall_up = true_up / up_count if up_count else 0.0
    recall_down = true_down / down_count if down_count else 0.0
    return {
        "samples": count,
        "accuracy": accuracy,
        "balanced_accuracy": (recall_up + recall_down) / 2.0,
        "train_majority_baseline_accuracy": (
            up_count / count if majority_class else down_count / count
        ),
        "up_outcome_rate": up_count / count,
        "confidence_55pct_coverage": confident_count / count,
        "confidence_55pct_accuracy": (
            confident_correct / confident_count if confident_count else None
        ),
        "confusion_matrix": {
            "true_down": true_down,
            "false_up": false_up,
            "false_down": false_down,
            "true_up": true_up,
        },
    }


def main():
    bars = load_bars()
    samples, scales = detect_samples(bars)
    split_samples_by_time, train_end, validation_end = split_samples(
        samples, len(bars)
    )
    packed = {name: pack(rows) for name, rows in split_samples_by_time.items()}
    if not packed["train"]["count"] or not packed["test"]["count"]:
        raise RuntimeError("Not enough event samples for chronological evaluation")

    means, feature_scales = standardize(packed["train"]["x"])
    weights, bias = train(packed["train"], means, feature_scales)
    train_up_rate = sum(packed["train"]["y"]) / packed["train"]["count"]
    majority_class = int(train_up_rate >= 0.5)
    metrics = {}
    for name in ("validation", "test"):
        if packed[name]["count"]:
            metrics[name] = evaluate(
                packed[name], weights, bias, means, feature_scales, majority_class
            )

    model = {
        "model": "standard-library logistic regression",
        "source": "HistData XAU/USD M1 resampled to M5",
        "feature_type": "variable-duration price swings and normalized swing paths; no indicators",
        "swing_detection": {
            "method": "directional change",
            "scale": "EWMA of M5 high-low range",
            "ewma_alpha": EWMA_ALPHA,
            "reversal_threshold_scale": SWING_SCALE,
        },
        "pattern_context": f"last {LEG_COUNT} completed swing legs; each leg has variable candle duration",
        "outcome": {
            "method": "first touch of upper or lower price barrier",
            "barrier_scale": OUTCOME_SCALE,
            "max_observation_bars": MAX_OUTCOME_BARS,
            "note": "The number of candles to form a pattern and resolve a label varies.",
        },
        "split": {
            "train_end_index": train_end,
            "validation_end_index": validation_end,
            "train_samples": packed["train"]["count"],
            "validation_samples": packed["validation"]["count"],
            "test_samples": packed["test"]["count"],
        },
        "data": {
            "bars": len(bars),
            "first_time_est": bars[0][0],
            "last_time_est": bars[-1][0],
        },
        "metrics": metrics,
        "normalization_means": means,
        "normalization_scales": feature_scales,
        "weights": weights,
        "bias": bias,
    }
    MODEL_PATH.write_text(json.dumps(model, indent=2))
    print(json.dumps({"model_path": str(MODEL_PATH), **model}, indent=2))


if __name__ == "__main__":
    main()
