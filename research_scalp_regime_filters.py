import datetime as dt
import json
from pathlib import Path

import backtest_trendline_trailing_stop as backtest
import research_indicator_combination as indicators
import research_trendline_setups as setups
import train_swing_model as base


ROOT = Path(__file__).parent
DATA_PATH = setups.DATA_PATH
OUTPUT_PATH = ROOT / "xauusd_scalp_regime_filter_research.json"
MAX_HOLD = backtest.MAX_HOLD_BARS
STOP_CONFIG = {
    "initial_sl": 1.0,
    "trail_activation": 0.5,
    "trail_gap": 0.5,
}
SIGNAL_RULE = "confirmed_support_breakdown"
MIN_TRAIN_TRADES = 80
COST_SCENARIOS_R = (0.0, 0.05, 0.10)


def build_filters():
    filters = {"no_filter": {"minimum_score": 0, "active_hours_only": False}}
    for score in range(1, 9):
        filters[f"bearish_alignment_at_least_{score}"] = {
            "minimum_score": score,
            "active_hours_only": False,
        }
    filters["active_hours_07_to_16_est"] = {
        "minimum_score": 0,
        "active_hours_only": True,
    }
    filters["alignment_3_and_active_hours"] = {
        "minimum_score": 3,
        "active_hours_only": True,
    }
    return filters


def calculate_alignment(bars, scales, market_features, event_index):
    market = market_features[event_index]
    timestamp = bars[event_index][0]
    source_time = dt.datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
    close = bars[event_index][4]
    scale = max(scales[event_index], 1e-8)
    one_hour_bearish = (
        close < bars[event_index - 12][4] if event_index >= 12 else False
    )
    four_hour_bearish = (
        close < bars[event_index - 48][4] if event_index >= 48 else False
    )
    features = market["indicators"]
    conditions = {
        "rsi_bearish": features[0] < 0.5,
        "macd_bearish": features[1] < 0.0,
        "macd_histogram_bearish": features[2] < 0.0,
        "below_ema20": features[3] < 0.0,
        "ema20_below_ema50": features[4] < 0.0,
        "below_bollinger_mid": features[5] < 0.5,
        "one_hour_return_bearish": one_hour_bearish,
        "four_hour_return_bearish": four_hour_bearish,
    }
    return {
        "conditions": conditions,
        "score": sum(conditions.values()),
        "hour": source_time.hour,
    }


def passes_filter(entry, filter_config):
    if entry["alignment_score"] < filter_config["minimum_score"]:
        return False
    if filter_config["active_hours_only"] and not (
        7 <= entry["hour"] < 17
    ):
        return False
    return True


def simulate_entries(entries, bars, scales, segment_end, filter_config):
    trades = []
    skipped = 0
    next_entry_index = 0
    for entry in entries:
        if not passes_filter(entry, filter_config):
            continue
        index = entry["event_index"]
        if index < next_entry_index:
            skipped += 1
            continue
        pnl, exit_index, reason = backtest.simulate_trade(
            bars,
            scales,
            index,
            -1,
            STOP_CONFIG,
            segment_end,
        )
        risk = STOP_CONFIG["initial_sl"] * max(scales[index], 1e-8)
        trades.append(
            {
                "R": pnl / risk,
                "exit_index": exit_index,
                "exit_reason": reason,
            }
        )
        next_entry_index = exit_index + 1
    return trades, skipped


def summarize(trades, skipped, costs):
    if not trades:
        return {
            "trades": 0,
            "skipped_while_open": skipped,
            "average_gross_R": None,
            "average_net_R": None,
        }
    values = [trade["R"] for trade in trades]
    net = [value - costs for value in values]
    wins = [value for value in net if value > 0.0]
    losses = [-value for value in net if value < 0.0]
    cumulative = peak = max_drawdown = 0.0
    for value in net:
        cumulative += value
        peak = max(peak, cumulative)
        max_drawdown = max(max_drawdown, peak - cumulative)
    return {
        "trades": len(trades),
        "skipped_while_open": skipped,
        "average_gross_R": sum(values) / len(values),
        "average_net_R": sum(net) / len(net),
        "total_net_R": sum(net),
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
    base.calculate_scales_cache = scales
    market_features = indicators.calculate_market_features(bars, scales)

    # Include all entry events; do not condition the backtest on the old
    # first-touch outcome label.
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

    entries = []
    for sample in samples:
        index = sample["event_index"]
        if index + MAX_HOLD >= len(bars):
            continue
        event = events.get(index)
        if not event or not event["rules"].get(SIGNAL_RULE, False):
            continue
        alignment = calculate_alignment(
            bars, scales, market_features, index
        )
        entries.append(
            {
                "event_index": index,
                "alignment_score": alignment["score"],
                "conditions": alignment["conditions"],
                "hour": alignment["hour"],
            }
        )

    split_entries = {}
    for split_name, (start, end) in bounds.items():
        split_entries[split_name] = [
            entry
            for entry in entries
            if entry["event_index"] >= start
            and entry["event_index"] + MAX_HOLD < end
        ]

    filters = build_filters()
    results = {}
    for filter_name, filter_config in filters.items():
        results[filter_name] = {
            "filter": filter_config,
            "splits": {},
        }
        for split_name, (start, end) in bounds.items():
            trades, skipped = simulate_entries(
                split_entries[split_name],
                bars,
                scales,
                end,
                filter_config,
            )
            results[filter_name]["splits"][split_name] = {
                f"round_trip_cost_{cost:g}R": summarize(trades, skipped, cost)
                for cost in COST_SCENARIOS_R
            }

    eligible = [
        (name, data)
        for name, data in results.items()
        if data["splits"]["train"]["round_trip_cost_0.05R"]["trades"]
        >= MIN_TRAIN_TRADES
    ]
    selected_name, selected = max(
        eligible,
        key=lambda item: item[1]["splits"]["train"][
            "round_trip_cost_0.05R"
        ]["average_net_R"],
    )
    payload = {
        "dataset": str(DATA_PATH),
        "bars": len(bars),
        "signal_events": len(entries),
        "signal_rule": SIGNAL_RULE,
        "trailing_stop": STOP_CONFIG,
        "bearish_alignment_score": [
            "RSI below 0.5",
            "MACD below zero",
            "MACD histogram below zero",
            "close below EMA20",
            "EMA20 below EMA50",
            "close below Bollinger middle",
            "prior 1-hour return negative",
            "prior 4-hour return negative",
        ],
        "active_hours_filter": "07:00-16:59 dataset fixed-EST time",
        "filter_selection": {
            "method": (
                "Choose the filter with highest training average net R at a "
                "hypothetical 0.05R round-trip cost; require at least "
                f"{MIN_TRAIN_TRADES} training trades."
            ),
            "selected_filter": selected_name,
            "selected_training_net_R": selected["splits"]["train"][
                "round_trip_cost_0.05R"
            ],
        },
        "results_by_filter": results,
        "execution_notes": {
            "take_profit": "None.",
            "entry": "Signal-bar close; stop checks begin on the next bar.",
            "maximum_hold_bars": MAX_HOLD,
            "position_management": "One open position for this setup at a time.",
            "costs": (
                "Round-trip cost is expressed as a sensitivity deduction in R, "
                "not a broker quote. Historical source is bid-only."
            ),
            "test_period": (
                "Diagnostic only; prior research has inspected this period."
            ),
        },
    }
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2))
    print(f"Saved {OUTPUT_PATH}")
    print(f"Filter selected on training only: {selected_name}")
    for filter_name, data in results.items():
        train = data["splits"]["train"]["round_trip_cost_0.05R"]
        validation = data["splits"]["validation"]["round_trip_cost_0.05R"]
        test = data["splits"]["test_diagnostic_only"]["round_trip_cost_0.05R"]
        print(
            f"{filter_name}: train n={train['trades']} avgNetR={train['average_net_R']}, "
            f"validation n={validation['trades']} avgNetR={validation['average_net_R']}, "
            f"test n={test['trades']} avgNetR={test['average_net_R']}"
        )


if __name__ == "__main__":
    main()
