import bisect
import json
from pathlib import Path

import backtest_m5_signal_m1_trailing as m1_backtest
import research_trendline_setups as setups
import train_swing_model as base


ROOT = Path(__file__).parent
DATA_PATH = ROOT / "XAUUSD_M5_HistData_2022-2026_clean.csv"
OUTPUT_PATH = ROOT / "xauusd_m1_trailing_stop_optimization.json"
SETUP = "support_rejection_uptrend"
SWING_SCALE = 0.75
MAX_M1_BARS_CHOICES = (15, 30, 60, 90)
INITIAL_STOPS = (0.5, 0.75, 1.0, 1.25)
TRAIL_ACTIVATIONS = (0.25, 0.5, 0.75, 1.0)
TRAIL_GAPS = (0.25, 0.5, 0.75)
MIN_TRAIN_TRADES = 100
SELECTION_COST_R = 0.02
COST_SCENARIOS_R = (0.0, 0.01, 0.02, 0.05, 0.10)


def configurations():
    for initial in INITIAL_STOPS:
        for activation in TRAIL_ACTIVATIONS:
            for gap in TRAIL_GAPS:
                for horizon in MAX_M1_BARS_CHOICES:
                    yield (
                        {
                            "initial_sl": initial,
                            "trail_activation": activation,
                            "trail_gap": gap,
                        },
                        horizon,
                    )


def summarize(trades, cost):
    if not trades:
        return {
            "trades": 0,
            "average_net_R": None,
            "total_net_R": 0.0,
            "net_profit_factor": None,
            "max_drawdown_net_R": 0.0,
        }
    values = [trade["R"] - cost for trade in trades]
    wins = [value for value in values if value > 0.0]
    losses = [-value for value in values if value < 0.0]
    cumulative = peak = max_drawdown = 0.0
    for value in values:
        cumulative += value
        peak = max(peak, cumulative)
        max_drawdown = max(max_drawdown, peak - cumulative)
    return {
        "trades": len(values),
        "average_net_R": sum(values) / len(values),
        "total_net_R": sum(values),
        "net_profit_factor": sum(wins) / sum(losses) if losses else None,
        "max_drawdown_net_R": max_drawdown,
    }


def evaluate(config, horizon, candidates, bounds, bars, m1_data, m1_split_ends):
    times, opens, highs, lows, closes = m1_data
    results = {}
    for split, (_, end) in bounds.items():
        m1_split_end = m1_split_ends[split]
        eligible = [
            item
            for item in candidates[split]
            if item["m1_start"] + horizon < m1_split_end
        ]
        trades = []
        next_entry_index = 0
        for item in eligible:
            if item["m1_start"] < next_entry_index:
                continue
            pnl, exit_index, reason = m1_backtest.simulate_m1_trade(
                times,
                opens,
                highs,
                lows,
                closes,
                item["m1_start"],
                item["entry_price"],
                item["scale"],
                1,
                m1_split_end,
                config=config,
                max_m1_bars=horizon,
            )
            risk = config["initial_sl"] * max(item["scale"], 1e-8)
            trades.append(
                {
                    "R": pnl / risk,
                    "exit_index": exit_index,
                    "exit_reason": reason,
                }
            )
            next_entry_index = exit_index + 1
        results[split] = {
            f"cost_{cost:g}R": summarize(trades, cost)
            for cost in COST_SCENARIOS_R
        }
    return results


def main():
    base.DATA_PATH = DATA_PATH
    setups.DATA_PATH = DATA_PATH
    base.SWING_SCALE = SWING_SCALE
    setups.SWING_SCALE = SWING_SCALE
    base.make_pattern_features = setups.collect_setups
    setups.events = {}
    bars = base.load_bars()
    m5_scales = base.calculate_scales(bars)
    base.first_barrier_outcome = (
        lambda _bars, _scales, start_index: (0, start_index + 1)
    )
    samples, _ = base.detect_samples(bars)
    events = setups.events

    times, opens, highs, lows, closes, duplicate_rows, conflict_rows, archives = (
        m1_backtest.load_m1_archives()
    )
    m1_data = (times, opens, highs, lows, closes)
    m5_times = [m1_backtest.parse_m5_minute(row[0]) for row in bars]
    train_end = int(len(bars) * 0.70)
    validation_end = int(len(bars) * 0.85)
    bounds = {
        "train": (0, train_end),
        "validation": (train_end, validation_end),
        "test_diagnostic_only": (validation_end, len(bars)),
    }
    candidates = {}
    m1_split_ends = {}
    for split, (start, end) in bounds.items():
        m1_split_end = (
            bisect.bisect_left(times, m5_times[end])
            if end < len(bars)
            else len(times)
        )
        m1_split_ends[split] = m1_split_end
        rows = []
        for sample in samples:
            index = sample["event_index"]
            if not start <= index < end:
                continue
            event = events.get(index)
            if not event or not event["rules"].get(SETUP, False):
                continue
            entry_minute = m5_times[index] + 5
            m1_start = bisect.bisect_left(times, entry_minute)
            if m1_start >= len(times) or times[m1_start] != entry_minute:
                continue
            rows.append(
                {
                    "m1_start": m1_start,
                    "entry_price": bars[index][4],
                    "scale": m5_scales[index],
                }
            )
        candidates[split] = rows

    all_results = []
    for config, horizon in configurations():
        split_results = evaluate(
            config,
            horizon,
            candidates,
            bounds,
            bars,
            m1_data,
            m1_split_ends,
        )
        train_stats = split_results["train"][f"cost_{SELECTION_COST_R:g}R"]
        if train_stats["trades"] < MIN_TRAIN_TRADES:
            continue
        all_results.append(
            {
                "config": config,
                "max_hold_m1_bars": horizon,
                "splits": split_results,
            }
        )

    if not all_results:
        raise RuntimeError("No M1 trailing-stop configuration met train minimum")
    all_results.sort(
        key=lambda item: (
            item["splits"]["train"][
                f"cost_{SELECTION_COST_R:g}R"
            ]["average_net_R"],
            item["splits"]["train"][
                f"cost_{SELECTION_COST_R:g}R"
            ]["trades"],
        ),
        reverse=True,
    )
    best = all_results[0]
    payload = {
        "dataset": str(DATA_PATH),
        "m1_source": "Existing local HistData archives",
        "m1_archives": archives,
        "m1_unique_bars": len(times),
        "m1_duplicate_rows_skipped": duplicate_rows,
        "m1_conflicting_timestamps_skipped": conflict_rows,
        "m5_bars": len(bars),
        "setup": SETUP,
        "swing_scale": SWING_SCALE,
        "selection_policy": {
            "selected_on": "train split only",
            "objective": (
                f"Highest average net R after a hypothetical "
                f"{SELECTION_COST_R}R round-trip cost"
            ),
            "minimum_train_trades": MIN_TRAIN_TRADES,
            "grid_size": (
                len(INITIAL_STOPS)
                * len(TRAIL_ACTIVATIONS)
                * len(TRAIL_GAPS)
                * len(MAX_M1_BARS_CHOICES)
            ),
        },
        "best_train_selected_configuration": best,
        "top_20_training_configurations": all_results[:20],
        "assumptions": {
            "take_profit": "None.",
            "entry": "M5 signal-bar close; stop checks begin on following M1 bar.",
            "max_holding": "15, 30, 60, or 90 M1 bars.",
            "position_management": "One open position for this setup.",
            "session_gap": "Exit at prior M1 close if the next M1 timestamp gaps.",
            "cost_scenarios": (
                "0 to 0.10R per round trip; sensitivity only, not a broker quote."
            ),
            "test_period": (
                "Diagnostic only; prior analysis has inspected this period."
            ),
        },
    }
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2))
    print(f"Saved {OUTPUT_PATH}")
    print("Best M1 trailing configuration selected on train:")
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
