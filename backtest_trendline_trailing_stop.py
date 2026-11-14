import json
import math
import statistics
from pathlib import Path

import research_trendline_setups as setups
import train_swing_model as base


ROOT = Path(__file__).parent
DATA_PATH = setups.DATA_PATH
OUTPUT_PATH = ROOT / "xauusd_trendline_trailing_stop_backtest.json"
MAX_HOLD_BARS = base.MAX_OUTCOME_BARS

# These are deliberately fixed before examining the held-out results.
STOP_CONFIGS = [
    {"initial_sl": initial, "trail_activation": activation, "trail_gap": gap}
    for initial in (1.0, 1.5)
    for activation in (0.5, 1.0)
    for gap in (0.5, 1.0)
]
STRATEGIES = {
    "resistance_rejection_downtrend": -1,
    "confirmed_resistance_breakout": 1,
    "confirmed_support_breakdown": -1,
    "support_rejection_uptrend": 1,
}


def simulate_trade(bars, scales, entry_index, direction, config, segment_end):
    entry = bars[entry_index][4]
    scale = max(scales[entry_index], 1e-8)
    risk = config["initial_sl"] * scale
    stop = entry - direction * risk
    best_price = entry
    activation = config["trail_activation"] * scale
    trail_gap = config["trail_gap"] * scale
    last_index = min(entry_index + MAX_HOLD_BARS, segment_end - 1)

    for index in range(entry_index + 1, last_index + 1):
        open_price, high, low, close = bars[index][1:5]
        if direction == 1:
            if open_price <= stop:
                exit_price = open_price
                return exit_price - entry, index, "stop"
            if low <= stop:
                return stop - entry, index, "stop"
            best_price = max(best_price, high)
            if best_price - entry >= activation:
                stop = max(stop, best_price - trail_gap)
        else:
            if open_price >= stop:
                exit_price = open_price
                return entry - exit_price, index, "stop"
            if high >= stop:
                return entry - stop, index, "stop"
            best_price = min(best_price, low)
            if entry - best_price >= activation:
                stop = min(stop, best_price + trail_gap)

    exit_index = last_index
    exit_price = bars[exit_index][4]
    pnl = direction * (exit_price - entry)
    return pnl, exit_index, "time"


def summarize(trades, skipped_signals):
    if not trades:
        return {
            "trades": 0,
            "skipped_while_position_open": skipped_signals,
            "win_rate": None,
            "average_R": None,
        }
    results = [trade["R"] for trade in trades]
    wins = [value for value in results if value > 0]
    losses = [-value for value in results if value < 0]
    cumulative = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for value in results:
        cumulative += value
        peak = max(peak, cumulative)
        max_drawdown = max(max_drawdown, peak - cumulative)
    return {
        "trades": len(trades),
        "skipped_while_position_open": skipped_signals,
        "win_rate": len(wins) / len(trades),
        "average_R": statistics.fmean(results),
        "median_R": statistics.median(results),
        "total_R": sum(results),
        "gross_profit_factor": (
            sum(wins) / sum(losses) if losses else None
        ),
        "max_drawdown_R": max_drawdown,
        "stop_exit_rate": sum(trade["exit_reason"] == "stop" for trade in trades)
        / len(trades),
        "time_exit_rate": sum(trade["exit_reason"] == "time" for trade in trades)
        / len(trades),
    }


def main():
    base.DATA_PATH = DATA_PATH
    base.SWING_SCALE = setups.SWING_SCALE
    base.make_pattern_features = setups.collect_setups
    setups.events = {}

    bars = base.load_bars()
    scales = base.calculate_scales(bars)
    # Keep every detected entry event. The normal research labeler drops
    # events whose future barrier outcome is unresolved, which would bias a
    # stop-based backtest by filtering entries using future information.
    base.first_barrier_outcome = (
        lambda _bars, _scales, start_index: (0, start_index + 1)
    )
    samples, _ = base.detect_samples(bars)
    train_end = int(len(bars) * 0.70)
    validation_end = int(len(bars) * 0.85)
    split_bounds = {
        "train": (0, train_end),
        "validation": (train_end, validation_end),
        "test": (validation_end, len(bars)),
    }
    split_rows = {
        "train": [
            sample
            for sample in samples
            if sample["event_index"] + MAX_HOLD_BARS < train_end
        ],
        "validation": [
            sample
            for sample in samples
            if sample["event_index"] >= train_end
            and sample["event_index"] + MAX_HOLD_BARS < validation_end
        ],
        "test": [
            sample
            for sample in samples
            if sample["event_index"] >= validation_end
            and sample["event_index"] + MAX_HOLD_BARS < len(bars)
        ],
    }

    signal_rows = {}
    for split_name, rows in split_rows.items():
        start, end = split_bounds[split_name]
        candidates = []
        for sample in rows:
            event_index = sample["event_index"]
            if event_index + MAX_HOLD_BARS >= end:
                continue
            data = setups.events.get(event_index)
            if data is None:
                continue
            for rule_name, direction in STRATEGIES.items():
                if data["rules"].get(rule_name, False):
                    candidates.append(
                        {
                            "event_index": event_index,
                            "direction": direction,
                            "rule": rule_name,
                        }
                    )
        signal_rows[split_name] = candidates

    results = {}
    for rule_name in STRATEGIES:
        results[rule_name] = {}
        for config in STOP_CONFIGS:
            config_name = (
                f"sl{config['initial_sl']}_"
                f"activate{config['trail_activation']}_"
                f"gap{config['trail_gap']}"
            )
            config_results = {}
            for split_name, (start, end) in split_bounds.items():
                signals = [
                    row
                    for row in signal_rows.get(split_name, [])
                    if row["rule"] == rule_name
                ]
                trades = []
                skipped = 0
                available_from = start
                for signal in signals:
                    entry_index = signal["event_index"]
                    if entry_index < available_from:
                        skipped += 1
                        continue
                    pnl, exit_index, exit_reason = simulate_trade(
                        bars,
                        scales,
                        entry_index,
                        signal["direction"],
                        config,
                        end,
                    )
                    initial_risk = (
                        config["initial_sl"] * max(scales[entry_index], 1e-8)
                    )
                    trades.append(
                        {
                            "R": pnl / initial_risk,
                            "exit_index": exit_index,
                            "exit_reason": exit_reason,
                        }
                    )
                    available_from = exit_index + 1
                display_name = (
                    "test_diagnostic_only" if split_name == "test" else split_name
                )
                config_results[display_name] = summarize(trades, skipped)
            results[rule_name][config_name] = {
                "parameters": config,
                "splits": config_results,
            }

    payload = {
        "dataset": str(DATA_PATH),
        "bars": len(bars),
        "detected_pattern_events": len(samples),
        "split_indices": {
            "train_end_index": train_end,
            "validation_end_index": validation_end,
        },
        "execution_assumptions": {
            "entry": "At signal-bar close; stop checks begin on the next bar.",
            "take_profit": "None.",
            "initial_stop": "Entry plus/minus the configured multiple of EWMA M5 range.",
            "trailing_stop": (
                "Activates after the configured favorable move and follows the "
                "best observed price by the configured range multiple. Stop "
                "updates become active from the following candle."
            ),
            "intrabar_ambiguity": (
                "Existing stop is checked before updating the trail from the "
                "current candle's high/low; this avoids assuming a favorable "
                "intrabar order when OHLC does not reveal the path."
            ),
            "position_management": (
                "One position per setup strategy; later signals are ignored "
                "until exit. Maximum hold is 144 M5 bars, then exit at close."
            ),
            "costs": "HistData is bid-only; spread, commission, and slippage are excluded.",
        },
        "strategies": results,
        "warning": (
            "Stop parameters are a research grid, not optimized live settings. "
            "Validation and test results are exploratory; test period was "
            "previously inspected. Gross R is not net profitability."
        ),
    }
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2))
    print(f"Saved {OUTPUT_PATH}")
    for rule_name, configs in results.items():
        print(f"\n{rule_name}")
        for config_name, data in configs.items():
            val = data["splits"]["validation"]
            test = data["splits"]["test_diagnostic_only"]
            print(
                f"  {config_name}: "
                f"validation n={val['trades']} avgR={val['average_R']}, "
                f"test n={test['trades']} avgR={test['average_R']}"
            )


if __name__ == "__main__":
    main()
