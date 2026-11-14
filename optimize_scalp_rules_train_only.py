import datetime as dt
import json
from pathlib import Path

import backtest_trendline_trailing_stop as backtest
import research_indicator_combination as indicators
import research_trendline_setups as setups
import train_swing_model as base


ROOT = Path(__file__).parent
DATA_PATH = setups.DATA_PATH
OUTPUT_PATH = ROOT / "xauusd_scalp_train_selected_rules.json"
SIGNAL_DIRECTIONS = {
    "resistance_rejection_downtrend": -1,
    "confirmed_resistance_breakout": 1,
    "confirmed_support_breakdown": -1,
    "support_rejection_uptrend": 1,
}
STOP_CONFIGS = backtest.STOP_CONFIGS
MIN_TRAIN_TRADES = 75
COST_FOR_SELECTION_R = 0.05
COST_SCENARIOS_R = (0.0, 0.02, 0.05, 0.10)


def aligned_score(market, direction, one_hour_return, four_hour_return):
    features = market["indicators"]
    if direction == 1:
        return sum(
            (
                features[0] > 0.5,
                features[1] > 0.0,
                features[2] > 0.0,
                features[3] > 0.0,
                features[4] > 0.0,
                features[5] > 0.5,
            )
        ) + int(one_hour_return > 0.0) + int(
            four_hour_return > 0.0
        )
    return sum(
        (
            features[0] < 0.5,
            features[1] < 0.0,
            features[2] < 0.0,
            features[3] < 0.0,
            features[4] < 0.0,
            features[5] < 0.5,
        )
    ) + int(one_hour_return < 0.0) + int(
        four_hour_return < 0.0
    )


def filters():
    return [
        {"minimum_alignment": score, "active_hours_only": active}
        for score in range(9)
        for active in (False, True)
    ]


def passes(entry, config):
    if entry["alignment"] < config["minimum_alignment"]:
        return False
    if config["active_hours_only"] and not (7 <= entry["hour"] < 17):
        return False
    return True


def summarize(trades, cost):
    if not trades:
        return {
            "trades": 0,
            "average_net_R": None,
            "total_net_R": 0.0,
            "net_profit_factor": None,
        }
    values = [trade["R"] - cost for trade in trades]
    wins = [value for value in values if value > 0.0]
    losses = [-value for value in values if value < 0.0]
    equity = peak = max_drawdown = 0.0
    for value in values:
        equity += value
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, peak - equity)
    return {
        "trades": len(values),
        "average_net_R": sum(values) / len(values),
        "total_net_R": sum(values),
        "net_profit_factor": sum(wins) / sum(losses) if losses else None,
        "max_drawdown_net_R": max_drawdown,
    }


def main():
    global events
    base.DATA_PATH = DATA_PATH
    base.SWING_SCALE = setups.SWING_SCALE
    base.make_pattern_features = setups.collect_setups
    setups.events = {}
    bars = base.load_bars()
    scales = base.calculate_scales(bars)
    market_features = indicators.calculate_market_features(bars, scales)

    # Keep every entry event; no future barrier label is used to qualify it.
    base.first_barrier_outcome = (
        lambda _bars, _scales, start_index: (0, start_index + 1)
    )
    samples, _ = base.detect_samples(bars)
    events = setups.events
    train_end = int(len(bars) * 0.70)
    validation_end = int(len(bars) * 0.85)
    bounds = {
        "train": (0, train_end),
        "validation": (train_end, validation_end),
        "test_diagnostic_only": (validation_end, len(bars)),
    }

    split_entries = {}
    for split_name, (start, end) in bounds.items():
        rows = []
        for sample in samples:
            index = sample["event_index"]
            if index < start or index + backtest.MAX_HOLD_BARS >= end:
                continue
            data = events.get(index)
            if data is None:
                continue
            for rule, direction in SIGNAL_DIRECTIONS.items():
                if not data["rules"].get(rule, False):
                    continue
                market = market_features[index]
                close = bars[index][4]
                one_hour_return = (
                    close - bars[index - 12][4] if index >= 12 else 0.0
                )
                four_hour_return = (
                    close - bars[index - 48][4] if index >= 48 else 0.0
                )
                rows.append(
                    {
                        "event_index": index,
                        "rule": rule,
                        "direction": direction,
                        "alignment": aligned_score(
                            market,
                            direction,
                            one_hour_return,
                            four_hour_return,
                        ),
                        "hour": dt.datetime.strptime(
                            bars[index][0], "%Y-%m-%d %H:%M:%S"
                        ).hour,
                    }
                )
        split_entries[split_name] = rows

    trade_cache = {}
    for rule, direction in SIGNAL_DIRECTIONS.items():
        trade_cache[rule] = {}
        for config in STOP_CONFIGS:
            config_name = (
                f"sl{config['initial_sl']}_"
                f"activate{config['trail_activation']}_"
                f"gap{config['trail_gap']}"
            )
            trade_cache[rule][config_name] = {}
            for split_name, (_, end) in bounds.items():
                cached = []
                for entry in split_entries[split_name]:
                    if entry["rule"] != rule:
                        continue
                    index = entry["event_index"]
                    pnl, exit_index, reason = backtest.simulate_trade(
                        bars, scales, index, direction, config, end
                    )
                    risk = config["initial_sl"] * max(scales[index], 1e-8)
                    cached.append(
                        {
                            **entry,
                            "R": pnl / risk,
                            "exit_index": exit_index,
                            "exit_reason": reason,
                        }
                    )
                trade_cache[rule][config_name][split_name] = cached

    candidates = []
    report = {}
    for rule in SIGNAL_DIRECTIONS:
        report[rule] = {}
        for config in STOP_CONFIGS:
            config_name = (
                f"sl{config['initial_sl']}_"
                f"activate{config['trail_activation']}_"
                f"gap{config['trail_gap']}"
            )
            report[rule][config_name] = {}
            for filter_config in filters():
                filter_name = (
                    f"alignment_{filter_config['minimum_alignment']}"
                    f"_hours_{int(filter_config['active_hours_only'])}"
                )
                split_stats = {}
                for split_name in bounds:
                    trades = []
                    next_entry_index = 0
                    for entry in trade_cache[rule][config_name][split_name]:
                        if not passes(entry, filter_config):
                            continue
                        if entry["event_index"] < next_entry_index:
                            continue
                        trades.append(entry)
                        next_entry_index = entry["exit_index"] + 1
                    split_stats[split_name] = {
                        f"cost_{cost:g}R": summarize(trades, cost)
                        for cost in COST_SCENARIOS_R
                    }
                report[rule][config_name][filter_name] = {
                    "filter": filter_config,
                    "splits": split_stats,
                }
                train_stats = split_stats["train"][
                    f"cost_{COST_FOR_SELECTION_R:g}R"
                ]
                if train_stats["trades"] >= MIN_TRAIN_TRADES:
                    candidates.append(
                        (
                            train_stats["average_net_R"],
                            train_stats["trades"],
                            rule,
                            config_name,
                            filter_name,
                        )
                    )

    if not candidates:
        raise RuntimeError("No training candidate met the minimum trade count")
    _, _, best_rule, best_config, best_filter = max(candidates)
    selected = report[best_rule][best_config][best_filter]
    payload = {
        "dataset": str(DATA_PATH),
        "bars": len(bars),
        "candidate_entry_events_by_split": {
            name: len(rows) for name, rows in split_entries.items()
        },
        "selection_policy": {
            "selection_data": "train split only",
            "objective": (
                f"Highest average net R after hypothetical {COST_FOR_SELECTION_R}R "
                "round-trip cost"
            ),
            "minimum_train_trades": MIN_TRAIN_TRADES,
            "selected": {
                "setup": best_rule,
                "stop_config": best_config,
                "filter": best_filter,
                "training_metrics": selected["splits"]["train"][
                    f"cost_{COST_FOR_SELECTION_R:g}R"
                ],
            },
        },
        "selected_strategy_results": selected["splits"],
        "all_training_selected_candidates": report,
        "assumptions": {
            "take_profit": "None.",
            "entry": "Signal-bar close; stop checks start next candle.",
            "active_hours": "07:00-16:59 fixed-EST dataset time.",
            "position_management": "One open position per setup.",
            "maximum_hold_bars": backtest.MAX_HOLD_BARS,
            "cost_model": (
                "Sensitivity deductions of 0, 0.02, 0.05, or 0.10 initial-risk R "
                "per round trip; not a broker-specific quote."
            ),
            "historical_test": (
                "Test segment is diagnostic only because it was inspected earlier."
            ),
        },
    }
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2))
    print(f"Saved {OUTPUT_PATH}")
    print(
        f"Training-selected: {best_rule}, {best_config}, {best_filter}"
    )
    for split, costs in selected["splits"].items():
        print(
            f"{split}: "
            + ", ".join(
                f"{cost:g}R n={costs[f'cost_{cost:g}R']['trades']} "
                f"avgNetR={costs[f'cost_{cost:g}R']['average_net_R']}"
                for cost in COST_SCENARIOS_R
            )
        )


if __name__ == "__main__":
    main()
