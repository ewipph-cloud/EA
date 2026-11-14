import json
from pathlib import Path

import backtest_trendline_trailing_stop as backtest
import research_trendline_setups as setups
import train_swing_model as base


ROOT = Path(__file__).parent
DATA_PATH = setups.DATA_PATH
OUTPUT_PATH = ROOT / "xauusd_true_scalp_horizon_research.json"
SWING_SCALES = (0.75, 1.0, 1.5)
MAX_HOLD_CHOICES = (6, 12, 24, 36)
INITIAL_STOPS = (0.75, 1.0, 1.5)
TRAIL_ACTIVATIONS = (0.25, 0.5, 1.0)
TRAIL_GAPS = (0.25, 0.5, 0.75)
MIN_TRAIN_TRADES = 75
COST_SCENARIOS_R = (0.0, 0.02, 0.05, 0.10)
SELECTION_COST_R = 0.05


def stop_configs():
    return [
        {
            "initial_sl": initial,
            "trail_activation": activation,
            "trail_gap": gap,
        }
        for initial in INITIAL_STOPS
        for activation in TRAIL_ACTIVATIONS
        for gap in TRAIL_GAPS
    ]


def summarize(trades, cost):
    if not trades:
        return {
            "trades": 0,
            "average_net_R": None,
            "total_net_R": 0.0,
            "net_profit_factor": None,
            "max_drawdown_net_R": 0.0,
        }
    returns = [trade["R"] - cost for trade in trades]
    wins = [value for value in returns if value > 0.0]
    losses = [-value for value in returns if value < 0.0]
    cumulative = peak = max_drawdown = 0.0
    for value in returns:
        cumulative += value
        peak = max(peak, cumulative)
        max_drawdown = max(max_drawdown, peak - cumulative)
    return {
        "trades": len(returns),
        "average_net_R": sum(returns) / len(returns),
        "total_net_R": sum(returns),
        "net_profit_factor": sum(wins) / sum(losses) if losses else None,
        "max_drawdown_net_R": max_drawdown,
    }


def evaluate_candidate(entries, bars, scales, bounds, setup, direction, config, horizon):
    old_horizon = backtest.MAX_HOLD_BARS
    backtest.MAX_HOLD_BARS = horizon
    split_results = {}
    try:
        for split, (start, end) in bounds.items():
            signals = [
                entry
                for entry in entries
                if entry["setup"] == setup
                and start <= entry["event_index"]
                and entry["event_index"] + horizon < end
            ]
            trades = []
            available_from = start
            for signal in signals:
                index = signal["event_index"]
                if index < available_from:
                    continue
                pnl, exit_index, reason = backtest.simulate_trade(
                    bars, scales, index, direction, config, end
                )
                risk = config["initial_sl"] * max(scales[index], 1e-8)
                trades.append(
                    {
                        "R": pnl / risk,
                        "exit_index": exit_index,
                        "exit_reason": reason,
                    }
                )
                available_from = exit_index + 1
            split_results[split] = {
                f"cost_{cost:g}R": summarize(trades, cost)
                for cost in COST_SCENARIOS_R
            }
    finally:
        backtest.MAX_HOLD_BARS = old_horizon
    return split_results


def compact_candidate(candidate):
    return {
        "setup": candidate["setup"],
        "swing_scale": candidate["swing_scale"],
        "max_hold_bars": candidate["max_hold_bars"],
        "stop": candidate["stop"],
        "training": candidate["splits"]["train"],
        "validation": candidate["splits"]["validation"],
        "test_diagnostic_only": candidate["splits"]["test_diagnostic_only"],
    }


def main():
    global events
    base.DATA_PATH = DATA_PATH
    base.make_pattern_features = setups.collect_setups
    bars = base.load_bars()
    scales = base.calculate_scales(bars)
    base.first_barrier_outcome = (
        lambda _bars, _scales, start_index: (0, start_index + 1)
    )
    train_end = int(len(bars) * 0.70)
    validation_end = int(len(bars) * 0.85)
    bounds = {
        "train": (0, train_end),
        "validation": (train_end, validation_end),
        "test_diagnostic_only": (validation_end, len(bars)),
    }

    entries = []
    counts_by_scale = {}
    for swing_scale in SWING_SCALES:
        base.SWING_SCALE = swing_scale
        setups.SWING_SCALE = swing_scale
        setups.events = {}
        samples, _ = base.detect_samples(bars)
        events = setups.events
        scale_counts = {name: 0 for name in backtest.STRATEGIES}
        for sample in samples:
            index = sample["event_index"]
            event = events.get(index)
            if event is None:
                continue
            for setup_name in backtest.STRATEGIES:
                if event["rules"].get(setup_name, False):
                    entries.append(
                        {
                            "swing_scale": swing_scale,
                            "event_index": index,
                            "setup": setup_name,
                        }
                    )
                    scale_counts[setup_name] += 1
        counts_by_scale[str(swing_scale)] = scale_counts

    candidates = []
    for swing_scale in SWING_SCALES:
        scale_entries = [
            entry for entry in entries if entry["swing_scale"] == swing_scale
        ]
        for setup, direction in backtest.STRATEGIES.items():
            for config in stop_configs():
                for horizon in MAX_HOLD_CHOICES:
                    split_results = evaluate_candidate(
                        scale_entries,
                        bars,
                        scales,
                        bounds,
                        setup,
                        direction,
                        config,
                        horizon,
                    )
                    training = split_results["train"][
                        f"cost_{SELECTION_COST_R:g}R"
                    ]
                    if training["trades"] < MIN_TRAIN_TRADES:
                        continue
                    candidates.append(
                        {
                            "setup": setup,
                            "swing_scale": swing_scale,
                            "max_hold_bars": horizon,
                            "stop": config,
                            "splits": split_results,
                        }
                    )

    if not candidates:
        raise RuntimeError("No scalping candidate met the training trade minimum")
    candidates.sort(
        key=lambda candidate: (
            candidate["splits"]["train"][
                f"cost_{SELECTION_COST_R:g}R"
            ]["average_net_R"],
            candidate["splits"]["train"][
                f"cost_{SELECTION_COST_R:g}R"
            ]["trades"],
        ),
        reverse=True,
    )

    best_by_setup = {}
    for setup in backtest.STRATEGIES:
        matching = [candidate for candidate in candidates if candidate["setup"] == setup]
        if matching:
            best_by_setup[setup] = compact_candidate(matching[0])

    payload = {
        "dataset": str(DATA_PATH),
        "bars": len(bars),
        "date_range": {
            "first_bar": bars[0][0],
            "last_bar": bars[-1][0],
        },
        "candidate_events_by_swing_scale": counts_by_scale,
        "scalp_horizon_minutes": {
            str(bars): bars * 5 for bars in MAX_HOLD_CHOICES
        },
        "selection_policy": {
            "selected_on": "train split only",
            "objective": (
                f"Highest average net R after a hypothetical "
                f"{SELECTION_COST_R}R round-trip cost"
            ),
            "minimum_train_trades": MIN_TRAIN_TRADES,
            "grid_size": (
                len(SWING_SCALES)
                * len(backtest.STRATEGIES)
                * len(stop_configs())
                * len(MAX_HOLD_CHOICES)
            ),
        },
        "best_candidate_selected_on_training": compact_candidate(candidates[0]),
        "best_candidate_per_setup": best_by_setup,
        "top_20_training_candidates": [
            compact_candidate(candidate) for candidate in candidates[:20]
        ],
        "assumptions": {
            "take_profit": "None.",
            "entry": "Signal-bar close; stop checks start next candle.",
            "max_hold": "6, 12, 24, or 36 M5 bars, then close if SL was not hit.",
            "position_management": "One open position per setup at a time.",
            "costs": (
                "Sensitivity deductions from 0 to 0.10 initial-risk R; "
                "not broker-specific. Bid-only data excludes actual spread."
            ),
            "test_period": (
                "Diagnostic only; prior analysis has inspected this period."
            ),
        },
    }
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2))
    print(f"Saved {OUTPUT_PATH}")
    print("Best selected on training only:")
    print(json.dumps(payload["best_candidate_selected_on_training"], indent=2))
    print("Best per setup (net R after 0.05R cost):")
    for setup, candidate in best_by_setup.items():
        print(
            setup,
            "scale", candidate["swing_scale"],
            "hold", candidate["max_hold_bars"],
            "stop", candidate["stop"],
            "train/val/test",
            *(
                candidate[split][f"cost_{SELECTION_COST_R:g}R"]["average_net_R"]
                for split in ("training", "validation", "test_diagnostic_only")
            ),
        )


if __name__ == "__main__":
    main()
