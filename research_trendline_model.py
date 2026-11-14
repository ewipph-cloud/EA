import json
import math
from pathlib import Path

import research_indicator_combination as research
import train_swing_model as base


ROOT = Path(__file__).parent
BASE_RESULT_PATH = ROOT / "xauusd_indicator_combination_research.json"
BASE_MODEL_PATH = ROOT / "xauusd_m5_combined_indicators_model.json"
MODEL_PATH = ROOT / "xauusd_m5_trendline_combined_model.json"
RESULT_PATH = ROOT / "xauusd_trendline_research.json"
TRENDLINE_FEATURES = 14
MIN_COVERAGE = 0.10
MIN_SIGNALS = 100


def fit_trendline(bars, pivots, confirmation_index, scale, is_high):
    turns = []
    first_position = max(1, len(pivots) - 6)
    for position in range(first_position, len(pivots)):
        previous_index, previous_price = pivots[position - 1]
        pivot_index, pivot_price = pivots[position]
        pivot_is_high = pivot_price > previous_price
        if pivot_is_high != is_high:
            continue

        indices = range(previous_index, pivot_index + 1)
        if is_high:
            extreme_index = max(indices, key=lambda index: bars[index][2])
            level = bars[extreme_index][2]
        else:
            extreme_index = min(indices, key=lambda index: bars[index][3])
            level = bars[extreme_index][3]
        turns.append((extreme_index, level))

    points = turns[-3:]
    if len(points) < 2:
        return (0.0, 0.0, 0.0, 0.0, 0.0)

    mean_x = sum(point[0] for point in points) / len(points)
    mean_y = sum(point[1] for point in points) / len(points)
    variance_x = sum((point[0] - mean_x) ** 2 for point in points)
    if variance_x <= 1e-12:
        return (0.0, 0.0, 0.0, 0.0, 0.0)
    slope = sum(
        (point[0] - mean_x) * (point[1] - mean_y) for point in points
    ) / variance_x
    intercept = mean_y - slope * mean_x
    projected = intercept + slope * confirmation_index
    residual = math.sqrt(
        sum(
            (price - (intercept + slope * index)) ** 2
            for index, price in points
        )
        / len(points)
    )
    close = bars[confirmation_index][4]
    scale = max(scale, 1e-8)
    if is_high:
        penetration = bars[confirmation_index][2] - projected
    else:
        penetration = projected - bars[confirmation_index][3]
    return (
        max(-5.0, min(5.0, slope * 12.0 / scale)),
        max(-8.0, min(8.0, (close - projected) / scale)),
        max(-8.0, min(8.0, penetration / scale)),
        min(8.0, residual / scale),
        len(points) / 3.0,
    )


def add_trendline_features(bars, scales, pivots, confirmation_index):
    features = research.add_base_context(
        bars, scales, pivots, confirmation_index
    )
    if features is None:
        return None

    scale = max(scales[confirmation_index], 1e-8)
    support = fit_trendline(
        bars, pivots, confirmation_index, scale, is_high=False
    )
    resistance = fit_trendline(
        bars, pivots, confirmation_index, scale, is_high=True
    )
    market = research.market_features[confirmation_index]
    candle = market["candle"]
    indicators = market["indicators"]
    interactions = market["interactions"]
    features.extend(candle)
    features.extend(indicators)
    features.extend(interactions)
    features.extend(support)
    features.extend(resistance)
    features.extend(
        (
            max(-8.0, min(8.0, support[1] * (indicators[5] - 0.5))),
            max(-8.0, min(8.0, resistance[1] * (indicators[5] - 0.5))),
            max(-8.0, min(8.0, support[0] * indicators[3])),
            max(-8.0, min(8.0, resistance[0] * indicators[3])),
        )
    )
    return features


def get_probabilities(split, weights, bias, means, scales):
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


def metrics(split, probabilities, threshold):
    selected = correct = 0
    for probability, actual in zip(probabilities, split["y"]):
        if max(probability, 1.0 - probability) < threshold:
            continue
        selected += 1
        correct += int((probability >= 0.5) == actual)
    return {
        "signals": selected,
        "coverage": selected / split["count"] if split["count"] else 0.0,
        "accuracy": correct / selected if selected else None,
        "wilson_95_lower": research.wilson_lower(correct, selected),
    }


def threshold_for_coverage(split, probabilities, coverage):
    minimum = max(1, math.ceil(split["count"] * coverage))
    confidences = sorted(
        (max(value, 1.0 - value) for value in probabilities),
        reverse=True,
    )
    return confidences[minimum - 1]


def main():
    global original_pattern_features
    base.DATA_PATH = research.DATA_PATH
    base.SWING_SCALE = research.SWING_SCALE
    base.LEG_COUNT = research.LEG_COUNT

    bars = base.load_bars()
    range_scales = base.calculate_scales(bars)
    original_pattern_features = base.make_pattern_features
    research.original_pattern_features = original_pattern_features
    research.market_features = research.calculate_market_features(
        bars, range_scales
    )
    base.make_pattern_features = add_trendline_features
    base.DIMENSIONS = (
        base.FEATURES_PER_LEG * research.LEG_COUNT
        + 2
        + research.CONTEXT_FEATURES
        + 4
        + 9
        + 8
        + TRENDLINE_FEATURES
    )

    samples, _ = base.detect_samples(bars)
    split_rows, train_end, validation_end = base.split_samples(
        samples, len(bars)
    )
    packed = {name: base.pack(rows) for name, rows in split_rows.items()}
    means, scales = base.standardize(packed["train"]["x"])
    weights, bias = base.train(packed["train"], means, scales)

    validation_probabilities = get_probabilities(
        packed["validation"], weights, bias, means, scales
    )
    minimum = max(
        MIN_SIGNALS,
        math.ceil(packed["validation"]["count"] * MIN_COVERAGE),
    )
    choices = []
    for step in range(50, 100):
        threshold = step / 100.0
        result = metrics(
            packed["validation"], validation_probabilities, threshold
        )
        if result["signals"] >= minimum:
            choices.append(
                (result["accuracy"], result["coverage"], threshold, result)
            )
    if not choices:
        raise RuntimeError("No validation threshold meets minimum coverage")
    _, _, threshold, validation_metrics = max(choices)

    test_probabilities = get_probabilities(
        packed["test"], weights, bias, means, scales
    )
    test_metrics = metrics(packed["test"], test_probabilities, threshold)
    baseline = json.loads(BASE_RESULT_PATH.read_text())["models"][
        "pattern_context_indicators_interactions"
    ]
    baseline_model = json.loads(BASE_MODEL_PATH.read_text())
    matched_coverage = {}
    baseline_dimensions = base.DIMENSIONS
    for name in ("validation", "test"):
        baseline_rows = [
            {**sample, "features": sample["features"][:-TRENDLINE_FEATURES]}
            for sample in split_rows[name]
        ]
        baseline_split = base.pack(baseline_rows)
        base.DIMENSIONS = len(baseline_model["weights"])
        baseline_probabilities = get_probabilities(
            baseline_split,
            baseline_model["weights"],
            baseline_model["bias"],
            baseline_model["normalization_means"],
            baseline_model["normalization_scales"],
        )
        base.DIMENSIONS = baseline_dimensions
        matched_coverage[name] = {
            "baseline_probabilities": baseline_probabilities,
            "trendline_probabilities": (
                validation_probabilities if name == "validation" else test_probabilities
            ),
            "baseline_split": baseline_split,
            "trendline_split": packed[name],
        }

    baseline_validation = matched_coverage["validation"]
    baseline_threshold = threshold_for_coverage(
        baseline_validation["baseline_split"],
        baseline_validation["baseline_probabilities"],
        MIN_COVERAGE,
    )
    trendline_threshold = threshold_for_coverage(
        baseline_validation["trendline_split"],
        baseline_validation["trendline_probabilities"],
        MIN_COVERAGE,
    )
    equal_coverage = {
        "target_validation_coverage": MIN_COVERAGE,
        "baseline_threshold": baseline_threshold,
        "trendline_threshold": trendline_threshold,
        "baseline": {
            "validation": metrics(
                baseline_validation["baseline_split"],
                baseline_validation["baseline_probabilities"],
                baseline_threshold,
            ),
            "test_diagnostic_only": metrics(
                matched_coverage["test"]["baseline_split"],
                matched_coverage["test"]["baseline_probabilities"],
                baseline_threshold,
            ),
        },
        "trendline": {
            "validation": metrics(
                baseline_validation["trendline_split"],
                baseline_validation["trendline_probabilities"],
                trendline_threshold,
            ),
            "test_diagnostic_only": metrics(
                matched_coverage["test"]["trendline_split"],
                matched_coverage["test"]["trendline_probabilities"],
                trendline_threshold,
            ),
        },
    }
    result = {
        "dataset": str(research.DATA_PATH),
        "event_count": len(samples),
        "feature_dimensions": base.DIMENSIONS,
        "trendline_definition": (
            "Least-squares lines from up to three confirmed swing highs/lows. "
            "Levels use wick extremes within completed swing legs; features "
            "are evaluated only at the pivot confirmation bar."
        ),
        "split": {
            "train_end_index": train_end,
            "validation_end_index": validation_end,
            "train_samples": packed["train"]["count"],
            "validation_samples": packed["validation"]["count"],
            "test_samples": packed["test"]["count"],
        },
        "baseline_without_explicit_trendlines": {
            "validation": baseline["validation"],
            "test_diagnostic_only": baseline["test_diagnostic_only"],
        },
        "trendline_model": {
            "threshold": threshold,
            "validation": validation_metrics,
            "test_diagnostic_only": test_metrics,
        },
        "equal_validation_coverage_comparison": equal_coverage,
        "warning": (
            "The recent test period was inspected previously. Treat its "
            "metrics as diagnostic, not as a clean final estimate."
        ),
    }
    model = {
        "model": (
            "Logistic regression on variable swing patterns, candle morphology, "
            "combined indicators, and confirmed trendlines"
        ),
        "source": "HistData XAU/USD M1 resampled to M5",
        "trendline_definition": result["trendline_definition"],
        "validation_threshold": threshold,
        "normalization_means": means,
        "normalization_scales": scales,
        "weights": weights,
        "bias": bias,
        "split": result["split"],
    }
    MODEL_PATH.write_text(json.dumps(model, indent=2))
    RESULT_PATH.write_text(json.dumps(result, indent=2))
    print(json.dumps({"result_path": str(RESULT_PATH), "model_path": str(MODEL_PATH), **result}, indent=2))


if __name__ == "__main__":
    main()
