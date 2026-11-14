import json
import math
import random
from pathlib import Path

import train_swing_model as base


ROOT = Path(__file__).parent
RESULT_PATH = ROOT / "xauusd_m5_pattern_neighbors.json"
CONFIGS = ((1.5, 3), (2.0, 4), (2.5, 5))
NEIGHBOR_COUNTS = (5, 9, 15, 31, 61)
REFERENCE_COUNT = 800
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


def make_reference(split, means, scales, seed):
    count = split["count"]
    indices = list(range(count))
    random.Random(seed).shuffle(indices)
    indices = indices[: min(REFERENCE_COUNT, count)]
    references = []
    labels = []
    for sample in indices:
        offset = sample * base.DIMENSIONS
        references.append(
            [
                (split["x"][offset + feature] - means[feature]) / scales[feature]
                for feature in range(base.DIMENSIONS)
            ]
        )
        labels.append(int(split["y"][sample]))
    return references, labels


def predict_probabilities(split, references, reference_labels, means, scales):
    x = split["x"]
    y = split["y"]
    probabilities = {k: [] for k in NEIGHBOR_COUNTS}
    largest_k = max(NEIGHBOR_COUNTS)
    for sample in range(split["count"]):
        offset = sample * base.DIMENSIONS
        query = [
            (x[offset + feature] - means[feature]) / scales[feature]
            for feature in range(base.DIMENSIONS)
        ]
        distances = []
        for reference, label in zip(references, reference_labels):
            distance = 0.0
            for left, right in zip(query, reference):
                difference = left - right
                distance += difference * difference
            distances.append((distance, label))
        distances.sort(key=lambda item: item[0])

        positive_count = 0
        next_k_index = 0
        for rank, (_, label) in enumerate(distances[:largest_k], start=1):
            positive_count += label
            if next_k_index < len(NEIGHBOR_COUNTS) and rank == NEIGHBOR_COUNTS[next_k_index]:
                k = NEIGHBOR_COUNTS[next_k_index]
                probabilities[k].append(positive_count / k)
                next_k_index += 1
    return probabilities


def selective_metrics(probabilities, labels, threshold):
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


def select_validation_rule(probabilities_by_k, labels):
    minimum = max(MIN_SIGNALS, math.ceil(len(labels) * MIN_COVERAGE))
    candidates = []
    for k, probabilities in probabilities_by_k.items():
        for step in range(50, 100):
            threshold = step / 100.0
            metrics = selective_metrics(probabilities, labels, threshold)
            if metrics["signals"] >= minimum:
                candidates.append(
                    (
                        metrics["accuracy"],
                        metrics["wilson_95_lower"],
                        metrics["coverage"],
                        k,
                        threshold,
                        metrics,
                    )
                )
    if not candidates:
        return None
    _, lower_bound, _, k, threshold, metrics = max(candidates)
    return {
        "neighbors": k,
        "threshold": threshold,
        **metrics,
        "wilson_95_lower": lower_bound,
        "minimum_signals_required": minimum,
    }


def main():
    bars = base.load_bars()
    candidates = []
    for seed, (swing_scale, leg_count) in enumerate(CONFIGS, start=1):
        base.SWING_SCALE = swing_scale
        base.LEG_COUNT = leg_count
        base.DIMENSIONS = leg_count * base.FEATURES_PER_LEG + 2
        samples, _ = base.detect_samples(bars)
        split_rows, train_end, validation_end = base.split_samples(samples, len(bars))
        packed = {name: base.pack(rows) for name, rows in split_rows.items()}
        if not packed["train"]["count"] or not packed["validation"]["count"]:
            continue

        means, scales = base.standardize(packed["train"]["x"])
        references, reference_labels = make_reference(
            packed["train"], means, scales, seed
        )
        validation_probabilities = predict_probabilities(
            packed["validation"], references, reference_labels, means, scales
        )
        selection = select_validation_rule(
            validation_probabilities, packed["validation"]["y"]
        )
        if selection is None:
            continue

        candidates.append(
            {
                "swing_scale": swing_scale,
                "leg_count": leg_count,
                "dimensions": base.DIMENSIONS,
                "train_samples": packed["train"]["count"],
                "validation_samples": packed["validation"]["count"],
                "test_samples": packed["test"]["count"],
                "train_end_index": train_end,
                "validation_end_index": validation_end,
                "selection": selection,
                "means": means,
                "scales": scales,
                "references": references,
                "reference_labels": reference_labels,
                "test_split": packed["test"],
            }
        )

    if not candidates:
        raise RuntimeError("No pattern-neighbor candidate had sufficient validation data")
    candidates.sort(
        key=lambda item: (
            item["selection"]["wilson_95_lower"],
            item["selection"]["coverage"],
        ),
        reverse=True,
    )
    selected = candidates[0]
    base.LEG_COUNT = selected["leg_count"]
    base.DIMENSIONS = selected["dimensions"]
    test_probabilities = predict_probabilities(
        selected["test_split"],
        selected["references"],
        selected["reference_labels"],
        selected["means"],
        selected["scales"],
    )[selected["selection"]["neighbors"]]
    test_metrics = selective_metrics(
        test_probabilities,
        selected["test_split"]["y"],
        selected["selection"]["threshold"],
    )

    result = {
        "model": "nearest historical variable-duration swing patterns",
        "selection_rule": "Choose swing resolution, neighbor count, and confidence threshold on validation only; maximize accuracy subject to at least 10% coverage",
        "goal": "At least 80% accuracy on selective signals with at least 10% coverage",
        "target_definition": {
            "outcome": "first touch of either volatility-scaled price barrier",
            "barrier_scale": base.OUTCOME_SCALE,
            "max_observation_bars": base.MAX_OUTCOME_BARS,
        },
        "validation_best": {
            "swing_scale": selected["swing_scale"],
            "leg_count": selected["leg_count"],
            "train_samples": selected["train_samples"],
            "validation_samples": selected["validation_samples"],
            "test_samples": selected["test_samples"],
            "selection": selected["selection"],
        },
        "test_once": test_metrics,
        "candidate_count": len(candidates),
        "top_validation_candidates": [
            {
                "swing_scale": item["swing_scale"],
                "leg_count": item["leg_count"],
                "selection": item["selection"],
            }
            for item in candidates[:3]
        ],
        "warning": "Test period was previously inspected for an earlier model; this result is diagnostic, not a pristine confirmatory result.",
        "deployment_artifact": {
            "swing_scale": selected["swing_scale"],
            "leg_count": selected["leg_count"],
            "neighbors": selected["selection"]["neighbors"],
            "confidence_threshold": selected["selection"]["threshold"],
            "normalization_means": selected["means"],
            "normalization_scales": selected["scales"],
            "reference_patterns": selected["references"],
            "reference_labels": selected["reference_labels"],
        },
    }
    RESULT_PATH.write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            {
                "result_path": str(RESULT_PATH),
                "candidate_count": result["candidate_count"],
                "validation_best": result["validation_best"],
                "test_once": result["test_once"],
                "top_validation_candidates": result["top_validation_candidates"],
                "warning": result["warning"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
