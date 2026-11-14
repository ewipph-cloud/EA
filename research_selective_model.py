import json
import math
from pathlib import Path

import train_swing_model as base


ROOT = Path(__file__).parent
RESULT_PATH = ROOT / "xauusd_m5_selective_research.json"
SWING_SCALES = (1.5, 2.0, 2.5)
LEG_COUNTS = (3, 4, 5)
MIN_COVERAGE = 0.10
MIN_SIGNALS = 100


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


def collect_probabilities(split, weights, bias, means, scales):
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


def signal_metrics(probabilities, labels, threshold):
    selected = 0
    correct = 0
    for probability, label in zip(probabilities, labels):
        prediction = int(probability >= 0.5)
        if max(probability, 1.0 - probability) >= threshold:
            selected += 1
            correct += int(prediction == label)
    return {
        "signals": selected,
        "coverage": selected / len(labels) if labels else 0.0,
        "accuracy": correct / selected if selected else None,
        "wilson_95_lower": wilson_lower(correct, selected),
    }


def choose_threshold(probabilities, labels):
    minimum = max(MIN_SIGNALS, math.ceil(len(labels) * MIN_COVERAGE))
    candidates = []
    for step in range(50, 100):
        threshold = step / 100.0
        metrics = signal_metrics(probabilities, labels, threshold)
        if metrics["signals"] >= minimum:
            candidates.append(
                (
                    metrics["accuracy"],
                    metrics["wilson_95_lower"],
                    metrics["coverage"],
                    threshold,
                    metrics,
                )
            )
    if not candidates:
        return None
    _, lower_bound, _, threshold, metrics = max(candidates)
    return {
        "threshold": threshold,
        **metrics,
        "wilson_95_lower": lower_bound,
        "minimum_signals_required": minimum,
    }


def run_candidate(bars, swing_scale, leg_count):
    base.SWING_SCALE = swing_scale
    base.LEG_COUNT = leg_count
    base.DIMENSIONS = leg_count * base.FEATURES_PER_LEG + 2
    samples, _ = base.detect_samples(bars)
    split_rows, train_end, validation_end = base.split_samples(samples, len(bars))
    packed = {name: base.pack(rows) for name, rows in split_rows.items()}
    if not packed["train"]["count"] or not packed["validation"]["count"]:
        return None

    means, scales = base.standardize(packed["train"]["x"])
    weights, bias = base.train(packed["train"], means, scales)
    validation_probabilities = collect_probabilities(
        packed["validation"], weights, bias, means, scales
    )
    validation_selection = choose_threshold(
        validation_probabilities, packed["validation"]["y"]
    )
    if validation_selection is None:
        return None

    return {
        "swing_scale": swing_scale,
        "leg_count": leg_count,
        "dimensions": base.DIMENSIONS,
        "train_samples": packed["train"]["count"],
        "validation_samples": packed["validation"]["count"],
        "test_samples": packed["test"]["count"],
        "train_end_index": train_end,
        "validation_end_index": validation_end,
        "validation_selection": validation_selection,
        "weights": weights,
        "bias": bias,
        "means": means,
        "scales": scales,
        "test_split": packed["test"],
    }


def main():
    bars = base.load_bars()
    candidates = []
    for swing_scale in SWING_SCALES:
        for leg_count in LEG_COUNTS:
            candidate = run_candidate(bars, swing_scale, leg_count)
            if candidate is not None:
                candidates.append(candidate)

    if not candidates:
        raise RuntimeError("No candidate had enough validation signals")

    candidates.sort(
        key=lambda item: (
            item["validation_selection"]["accuracy"],
            item["validation_selection"]["coverage"],
            item["validation_selection"]["wilson_95_lower"],
        ),
        reverse=True,
    )
    selected = candidates[0]
    base.LEG_COUNT = selected["leg_count"]
    base.DIMENSIONS = selected["dimensions"]
    test_probabilities = collect_probabilities(
        selected["test_split"],
        selected["weights"],
        selected["bias"],
        selected["means"],
        selected["scales"],
    )
    test_metrics = signal_metrics(
        test_probabilities,
        selected["test_split"]["y"],
        selected["validation_selection"]["threshold"],
    )

    result = {
        "goal": "At least 80% accuracy on selective signals with at least 10% coverage",
        "selection_rule": "Choose configuration and confidence threshold using validation only; maximize accuracy subject to at least 10% coverage; report Wilson confidence bound",
        "target_definition": {
            "outcome": "first touch of either volatility-scaled price barrier",
            "barrier_scale": base.OUTCOME_SCALE,
            "max_observation_bars": base.MAX_OUTCOME_BARS,
        },
        "validation_best": {
            key: value
            for key, value in selected.items()
            if key not in ("weights", "bias", "means", "scales", "test_split")
        },
        "test_once": test_metrics,
        "candidate_count": len(candidates),
        "top_validation_candidates": [
            {
                "swing_scale": item["swing_scale"],
                "leg_count": item["leg_count"],
                "validation_selection": item["validation_selection"],
            }
            for item in candidates[:5]
        ],
        "warning": "Test period was previously inspected for the earlier baseline; this is a diagnostic holdout, not a pristine confirmatory result.",
        "model": {
            "feature_type": "variable-duration swing paths; no indicator signals",
            "swing_scale": selected["swing_scale"],
            "leg_count": selected["leg_count"],
            "confidence_threshold": selected["validation_selection"]["threshold"],
            "normalization_means": selected["means"],
            "normalization_scales": selected["scales"],
            "weights": selected["weights"],
            "bias": selected["bias"],
        },
    }
    RESULT_PATH.write_text(json.dumps(result, indent=2))
    print(json.dumps({"result_path": str(RESULT_PATH), **result}, indent=2))


if __name__ == "__main__":
    main()
