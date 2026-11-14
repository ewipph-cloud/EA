import datetime as dt
import json
import math
from pathlib import Path

import train_swing_model as base


ROOT = Path(__file__).parent
DATA_PATH = ROOT / "XAUUSD_M5_HistData_2022-2026.csv"
RESULT_PATH = ROOT / "xauusd_indicator_combination_research.json"
MODEL_PATH = ROOT / "xauusd_m5_combined_indicators_model.json"
SWING_SCALE = 1.5
LEG_COUNT = 3
CONTEXT_FEATURES = 6
MIN_COVERAGE = 0.10
MIN_SIGNALS = 100


def ema(values, period):
    alpha = 2.0 / (period + 1.0)
    result = []
    current = values[0]
    for value in values:
        current = alpha * value + (1.0 - alpha) * current
        result.append(current)
    return result


def calculate_market_features(bars, range_scales):
    closes = [bar[4] for bar in bars]
    ema_12 = ema(closes, 12)
    ema_20 = ema(closes, 20)
    ema_26 = ema(closes, 26)
    ema_50 = ema(closes, 50)
    macd = [fast - slow for fast, slow in zip(ema_12, ema_26)]
    macd_signal = ema(macd, 9)

    gains = [0.0]
    losses = [0.0]
    for previous, current in zip(closes, closes[1:]):
        difference = current - previous
        gains.append(max(difference, 0.0))
        losses.append(max(-difference, 0.0))

    bb_period = 20
    running_sum = running_squares = 0.0
    widths = []
    positions = []
    for index, close in enumerate(closes):
        running_sum += close
        running_squares += close * close
        if index >= bb_period:
            old = closes[index - bb_period]
            running_sum -= old
            running_squares -= old * old
        if index + 1 < bb_period:
            widths.append(0.0)
            positions.append(0.5)
            continue
        mean = running_sum / bb_period
        deviation = math.sqrt(
            max(running_squares / bb_period - mean * mean, 0.0)
        )
        band_width = 4.0 * deviation / max(abs(mean), 1e-8)
        widths.append(band_width)
        positions.append(
            0.5
            if deviation <= 1e-12
            else (close - (mean - 2.0 * deviation)) / (4.0 * deviation)
        )
    slow_width = ema(widths, 100)

    features = []
    for index, bar in enumerate(bars):
        start = max(1, index - 13)
        avg_gain = sum(gains[start : index + 1]) / max(index - start + 1, 1)
        avg_loss = sum(losses[start : index + 1]) / max(index - start + 1, 1)
        rsi = (
            0.5
            if avg_gain + avg_loss <= 1e-12
            else avg_gain / (avg_gain + avg_loss)
        )

        scale = max(range_scales[index], 1e-8)
        open_price, high, low, close = bar[1:5]
        candle_range = max(high - low, 1e-8)
        body = close - open_price
        upper_wick = high - max(open_price, close)
        lower_wick = min(open_price, close) - low
        candle_features = (
            body / candle_range,
            upper_wick / candle_range,
            lower_wick / candle_range,
            (close - low) / candle_range,
        )
        indicator_features = (
            rsi,
            macd[index] / scale,
            (macd[index] - macd_signal[index]) / scale,
            (close - ema_20[index]) / scale,
            (ema_20[index] - ema_50[index]) / scale,
            max(-1.0, min(2.0, positions[index])),
            widths[index],
            widths[index] / max(slow_width[index], 1e-8),
            candle_range / scale,
        )
        bb_position = indicator_features[5]
        trend = indicator_features[3]
        macd_scaled = indicator_features[1]
        histogram_scaled = indicator_features[2]
        interactions = (
            max(-4.0, min(4.0, trend * (bb_position - 0.5))),
            max(-4.0, min(4.0, trend * (rsi - 0.5))),
            max(-4.0, min(4.0, trend * macd_scaled)),
            max(-4.0, min(4.0, trend * histogram_scaled)),
            candle_features[2] * max(0.0, 0.2 - bb_position),
            candle_features[1] * max(0.0, bb_position - 0.8),
            max(-4.0, min(4.0, trend * indicator_features[7])),
            max(-4.0, min(4.0, trend * candle_features[0])),
        )
        features.append(
            {
                "candle": candle_features,
                "indicators": indicator_features,
                "interactions": interactions,
                "diagnostic": {
                    "bb_position": bb_position,
                    "rsi": rsi,
                    "trend": trend,
                    "bandwidth": widths[index],
                },
            }
        )
    return features


def add_base_context(bars, scales, pivots, confirmation_index):
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
        proportion * (1.0 - proportion) / count
        + z * z / (4.0 * count * count)
    )
    return (center - margin) / denominator


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
        "wilson_95_lower": wilson_lower(correct, selected),
    }


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


def error_groups(samples, probabilities, threshold, market_features):
    groups = {}
    for sample, probability in zip(samples, probabilities):
        if max(probability, 1.0 - probability) < threshold:
            continue
        index = sample["event_index"]
        details = market_features[index]["diagnostic"]
        bb = details["bb_position"]
        rsi = details["rsi"]
        trend = details["trend"]
        labels = (
            "BB: below 0.2" if bb < 0.2 else "BB: above 0.8" if bb > 0.8 else "BB: middle",
            "RSI: oversold" if rsi < 0.3 else "RSI: overbought" if rsi > 0.7 else "RSI: middle",
            "Trend: up" if trend > 0.25 else "Trend: down" if trend < -0.25 else "Trend: flat",
        )
        predicted = int(probability >= 0.5)
        for label in labels:
            row = groups.setdefault(label, {"signals": 0, "correct": 0})
            row["signals"] += 1
            row["correct"] += int(predicted == sample["label"])
    return {
        label: {
            **row,
            "accuracy": row["correct"] / row["signals"],
        }
        for label, row in sorted(groups.items())
        if row["signals"] >= 20
    }


def main():
    global original_pattern_features, market_features
    original_pattern_features = base.make_pattern_features
    base.DATA_PATH = DATA_PATH
    base.SWING_SCALE = SWING_SCALE
    base.LEG_COUNT = LEG_COUNT

    bars = base.load_bars()
    samples, range_scales = base.detect_samples(bars)
    market_features = calculate_market_features(bars, range_scales)
    base.make_pattern_features = add_base_context

    candidates = {
        "pattern_context": (),
        "pattern_context_candles": ("candle",),
        "pattern_context_indicators": ("indicators",),
        "pattern_context_candles_indicators": ("candle", "indicators"),
        "pattern_context_indicators_interactions": (
            "candle",
            "indicators",
            "interactions",
        ),
    }
    results = {}
    trained = {}
    for name, additions in candidates.items():
        extra_dimensions = sum(
            {"candle": 4, "indicators": 9, "interactions": 8}[addition]
            for addition in additions
        )
        base.DIMENSIONS = (
            LEG_COUNT * base.FEATURES_PER_LEG + 2 + CONTEXT_FEATURES
            + extra_dimensions
        )
        if additions:
            active_features = base.make_pattern_features

            def add_selected_features(
                bars, scales, pivots, confirmation_index, active=additions
            ):
                features = add_base_context(
                    bars, scales, pivots, confirmation_index
                )
                if features is None:
                    return None
                for addition in active:
                    features.extend(
                        market_features[confirmation_index][addition]
                    )
                return features

            base.make_pattern_features = add_selected_features
        else:
            base.make_pattern_features = add_base_context

        event_samples, _ = base.detect_samples(bars)
        split_rows, train_end, validation_end = base.split_samples(
            event_samples, len(bars)
        )
        packed = {key: base.pack(rows) for key, rows in split_rows.items()}
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
            raise RuntimeError(f"No valid confidence threshold for {name}")
        _, _, threshold, validation_metrics = max(choices)

        test_probabilities = get_probabilities(
            packed["test"], weights, bias, means, scales
        )
        test_metrics = metrics(packed["test"], test_probabilities, threshold)
        result = {
            "dimensions": base.DIMENSIONS,
            "threshold": threshold,
            "validation": validation_metrics,
            "test_diagnostic_only": test_metrics,
            "split": {
                "train_end_index": train_end,
                "validation_end_index": validation_end,
                "train_samples": packed["train"]["count"],
                "validation_samples": packed["validation"]["count"],
                "test_samples": packed["test"]["count"],
            },
        }
        if name == "pattern_context_indicators_interactions":
            result["validation_error_groups"] = error_groups(
                split_rows["validation"],
                validation_probabilities,
                threshold,
                market_features,
            )
            result["test_error_groups_diagnostic_only"] = error_groups(
                split_rows["test"],
                test_probabilities,
                threshold,
                market_features,
            )
            trained[name] = {
                "weights": weights,
                "bias": bias,
                "means": means,
                "scales": scales,
                "threshold": threshold,
            }
        results[name] = result

    final_model = {
        "model": "logistic regression on variable swing patterns, candle morphology, and combined indicators",
        "source": "HistData XAU/USD M1 resampled to M5",
        "features": {
            "base": [
                "three variable-duration swing legs",
                "UTC time and day-of-week cycles",
                "prior 1-hour and 4-hour returns",
            ],
            "candle": [
                "body-to-range",
                "upper-wick-to-range",
                "lower-wick-to-range",
                "close position in range",
            ],
            "indicators": [
                "RSI(14)",
                "MACD(12,26,9) and histogram, range-normalized",
                "EMA(20)/EMA(50) trend, range-normalized",
                "Bollinger %B(20,2), bandwidth, and relative bandwidth",
                "candle range relative to EWMA range",
            ],
            "interactions": [
                "trend conditional on Bollinger position and RSI",
                "trend conditional on MACD, bandwidth, and candle body",
                "upper/lower wick rejection conditional on Bollinger extremes",
            ],
        },
        "split": results["pattern_context_indicators_interactions"]["split"],
        "validation_threshold": trained["pattern_context_indicators_interactions"][
            "threshold"
        ],
        "normalization_means": trained["pattern_context_indicators_interactions"][
            "means"
        ],
        "normalization_scales": trained["pattern_context_indicators_interactions"][
            "scales"
        ],
        "weights": trained["pattern_context_indicators_interactions"]["weights"],
        "bias": trained["pattern_context_indicators_interactions"]["bias"],
    }
    MODEL_PATH.write_text(json.dumps(final_model, indent=2))
    payload = {
        "dataset": str(DATA_PATH),
        "bar_count": len(bars),
        "event_count": len(samples),
        "models": results,
        "warning": (
            "The recent test period was inspected in earlier experiments. "
            "Test metrics are diagnostic only, not a clean final estimate."
        ),
    }
    RESULT_PATH.write_text(json.dumps(payload, indent=2))
    print(json.dumps({"result_path": str(RESULT_PATH), "model_path": str(MODEL_PATH), **payload}, indent=2))


if __name__ == "__main__":
    main()
