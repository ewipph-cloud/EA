import json
from pathlib import Path

import backtest_trendline_trailing_stop as backtest
import research_trendline_setups as setups
import train_swing_model as base


ROOT = Path(__file__).parent
DATA_PATH = setups.DATA_PATH
OUTPUT_PATH = ROOT / "xauusd_idr_compounding_simulation.json"
STARTING_CAPITAL_IDR = 5_000_000.0
RISK_LEVELS = (0.005, 0.01, 0.02)
STRATEGY = "confirmed_support_breakdown"
CONFIG = {
    "initial_sl": 1.0,
    "trail_activation": 0.5,
    "trail_gap": 0.5,
}


def compound(trades, risk_fraction):
    equity = STARTING_CAPITAL_IDR
    peak = equity
    max_drawdown = 0.0
    wins = 0
    losses = 0
    for trade in trades:
        profit = equity * risk_fraction * trade["R"]
        equity += profit
        wins += int(profit > 0.0)
        losses += int(profit < 0.0)
        peak = max(peak, equity)
        if peak > 0.0:
            max_drawdown = max(max_drawdown, (peak - equity) / peak)
    return {
        "starting_capital_IDR": STARTING_CAPITAL_IDR,
        "ending_capital_IDR": round(equity, 2),
        "gross_profit_IDR": round(equity - STARTING_CAPITAL_IDR, 2),
        "opened_positions": len(trades),
        "winning_positions": wins,
        "losing_positions": losses,
        "max_drawdown_percent": round(max_drawdown * 100.0, 2),
    }


def main():
    base.DATA_PATH = DATA_PATH
    base.SWING_SCALE = setups.SWING_SCALE
    base.make_pattern_features = setups.collect_setups
    setups.events = {}

    bars = base.load_bars()
    scales = base.calculate_scales(bars)
    # Do not let the first-barrier labeler filter entry events based on future
    # price movement; trade outcomes come only from the stop simulation.
    base.first_barrier_outcome = (
        lambda _bars, _scales, start_index: (0, start_index + 1)
    )
    samples, _ = base.detect_samples(bars)

    train_end = int(len(bars) * 0.70)
    validation_end = int(len(bars) * 0.85)
    boundaries = {
        "train": (0, train_end),
        "validation": (train_end, validation_end),
        "test_diagnostic_only": (validation_end, len(bars)),
    }
    all_events = []
    for sample in samples:
        event_index = sample["event_index"]
        if event_index + backtest.MAX_HOLD_BARS >= len(bars):
            continue
        event = setups.events.get(event_index)
        if event and event["rules"].get(STRATEGY, False):
            all_events.append(
                {
                    "event_index": event_index,
                    "direction": -1,
                    "rule": STRATEGY,
                }
            )

    def create_trades(events, segment_end):
        trades = []
        next_entry_index = 0
        for event in events:
            entry_index = event["event_index"]
            if entry_index < next_entry_index:
                continue
            pnl, exit_index, exit_reason = backtest.simulate_trade(
                bars,
                scales,
                entry_index,
                event["direction"],
                CONFIG,
                segment_end,
            )
            initial_risk = CONFIG["initial_sl"] * max(
                scales[entry_index], 1e-8
            )
            trades.append(
                {
                    "R": pnl / initial_risk,
                    "entry_index": entry_index,
                    "exit_index": exit_index,
                    "exit_reason": exit_reason,
                }
            )
            next_entry_index = exit_index + 1
        return trades

    full_trades = create_trades(all_events, len(bars))
    split_trades = {}
    for name, (start, end) in boundaries.items():
        events = [
            event
            for event in all_events
            if event["event_index"] >= start
            and event["event_index"] + backtest.MAX_HOLD_BARS < end
        ]
        split_trades[name] = create_trades(events, end)

    results = {}
    for risk_fraction in RISK_LEVELS:
        risk_label = f"{risk_fraction * 100:g}%"
        results[risk_label] = {
            "full_history_compounded": compound(full_trades, risk_fraction),
            "periods_started_with_5M_IDR": {
                name: compound(trades, risk_fraction)
                for name, trades in split_trades.items()
            },
        }

    payload = {
        "dataset": str(DATA_PATH),
        "date_range": {
            "first_bar": bars[0][0],
            "last_bar": bars[-1][0],
        },
        "starting_capital_IDR": STARTING_CAPITAL_IDR,
        "position_sizing": (
            "Compounded: each position risks the selected percentage of current equity."
        ),
        "strategy": STRATEGY,
        "trailing_stop_configuration": CONFIG,
        "execution_assumptions": {
            "entry": "Signal-bar close; stop checks start on the next M5 bar.",
            "take_profit": "None.",
            "maximum_hold_bars": backtest.MAX_HOLD_BARS,
            "one_open_position_per_strategy": True,
            "spreads_commission_slippage_swap": (
                "Not included; source prices are bid-only. IDR results are "
                "risk-unit simulations and do not determine broker lot size."
            ),
            "test_period": (
                "Diagnostic only; this period was inspected in earlier research."
            ),
        },
        "results_by_risk": results,
    }
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2))
    print(f"Saved {OUTPUT_PATH}")
    for risk, data in results.items():
        full = data["full_history_compounded"]
        print(
            f"{risk}: full-history opens={full['opened_positions']}, "
            f"gross P/L=Rp{full['gross_profit_IDR']:,.0f}, "
            f"ending=Rp{full['ending_capital_IDR']:,.0f}, "
            f"max DD={full['max_drawdown_percent']:.2f}%"
        )
        for name, result in data["periods_started_with_5M_IDR"].items():
            print(
                f"  {name}: opens={result['opened_positions']}, "
                f"gross P/L=Rp{result['gross_profit_IDR']:,.0f}, "
                f"ending=Rp{result['ending_capital_IDR']:,.0f}, "
                f"max DD={result['max_drawdown_percent']:.2f}%"
            )


if __name__ == "__main__":
    main()
