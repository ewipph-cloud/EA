import bisect
import datetime as dt
import json
import math
from collections import Counter
from pathlib import Path

import backtest_m5_signal_m1_trailing as m1_data
import train_swing_model as base


ROOT = Path(__file__).parent
M5_PATH = ROOT / "XAUUSD_M5_HistData_2022-2026_clean.csv"
OUTPUT_PATH = ROOT / "xauusd_m1_entries_m5_context_relaxed.json"
M5_FAST_EMA = 5
M5_SLOW_EMA = 13
M5_BOLLINGER_LENGTH = 20
M5_SLOPE_LOOKBACK = 3
M1_ATR_ALPHA = 0.08
MIN_BODY_TO_RANGE = 0.0
STOP_ATR = 0.75
TRAIL_ACTIVATION_ATR = 0.25
TRAIL_GAP_ATR = 0.50
MAX_HOLD_M1_BARS = 5
COST_SCENARIOS_R = (0.0, 0.02, 0.05, 0.10)


def ema(values, period):
    alpha = 2.0 / (period + 1.0)
    result = []
    current = values[0]
    for value in values:
        current = alpha * value + (1.0 - alpha) * current
        result.append(current)
    return result


def rolling_bollinger_mid(values, length):
    result = [None] * len(values)
    window_sum = 0.0
    for index, value in enumerate(values):
        window_sum += value
        if index >= length:
            window_sum -= values[index - length]
        if index >= length - 1:
            result[index] = window_sum / length
    return result


def minute_atr(highs, lows):
    result = []
    current = max(highs[0] - lows[0], 1e-8)
    for high, low in zip(highs, lows):
        current = (
            M1_ATR_ALPHA * max(high - low, 1e-8)
            + (1.0 - M1_ATR_ALPHA) * current
        )
        result.append(current)
    return result


def daily_counts(trades, active_days):
    counts = Counter({day: 0 for day in active_days})
    counts.update(trade["day"] for trade in trades)
    if not active_days:
        return {
            "days_with_trades": 0,
            "mean_positions_per_active_day": 0.0,
            "median_positions_per_active_day": 0.0,
            "p10_positions_per_active_day": 0,
            "max_positions_per_active_day": 0,
            "active_days_at_least_100": 0,
            "active_day_count": 0,
        }
    active_days = sorted(counts.values())
    p10_index = max(0, math.ceil(len(active_days) * 0.10) - 1)
    return {
        "days_with_trades": sum(value > 0 for value in active_days),
        "mean_positions_per_active_day": sum(active_days) / len(active_days),
        "median_positions_per_active_day": (
            active_days[(len(active_days) - 1) // 2]
            + active_days[len(active_days) // 2]
        )
        / 2.0,
        "p10_positions_per_active_day": active_days[p10_index],
        "max_positions_per_active_day": active_days[-1],
        "active_days_at_least_100": sum(value >= 100 for value in active_days),
        "active_day_count": len(active_days),
    }


def summarize(trades, cost, active_days):
    if not trades:
        return {
            "positions": 0,
            "average_net_R": None,
            "total_net_R": 0.0,
            "profit_factor_net": None,
            **daily_counts(trades, active_days),
        }
    values = [trade["R"] - cost for trade in trades]
    wins = [value for value in values if value > 0.0]
    losses = [-value for value in values if value < 0.0]
    return {
        "positions": len(trades),
        "average_net_R": sum(values) / len(values),
        "total_net_R": sum(values),
        "profit_factor_net": sum(wins) / sum(losses) if losses else None,
        **daily_counts(trades, active_days),
    }


def build_m5_context(bars):
    closes = [bar[4] for bar in bars]
    fast = ema(closes, M5_FAST_EMA)
    slow = ema(closes, M5_SLOW_EMA)
    bb_mid = rolling_bollinger_mid(closes, M5_BOLLINGER_LENGTH)
    directions = [0] * len(bars)
    for index in range(M5_BOLLINGER_LENGTH - 1 + M5_SLOPE_LOOKBACK, len(bars)):
        if bb_mid[index] is None:
            continue
        rising = slow[index] > slow[index - M5_SLOPE_LOOKBACK]
        falling = slow[index] < slow[index - M5_SLOPE_LOOKBACK]
        if fast[index] > slow[index] and closes[index] > bb_mid[index] and rising:
            directions[index] = 1
        elif (
            fast[index] < slow[index]
            and closes[index] < bb_mid[index]
            and falling
        ):
            directions[index] = -1
    return directions


def simulate_trade(
    times, opens, highs, lows, closes, start, direction, atr, split_end
):
    entry = opens[start]
    risk = max(STOP_ATR * atr, 1e-8)
    stop = entry - direction * risk
    best = entry
    activation = TRAIL_ACTIVATION_ATR * atr
    gap = TRAIL_GAP_ATR * atr
    previous_close = entry
    previous_time = times[start] - 1
    end = min(start + MAX_HOLD_M1_BARS, len(times), split_end)

    for index in range(start, end):
        if times[index] - previous_time > 1:
            return direction * (previous_close - entry), index - 1, risk

        if direction == 1:
            if opens[index] <= stop:
                return opens[index] - entry, index, risk
            if lows[index] <= stop:
                return stop - entry, index, risk
            best = max(best, highs[index])
            if best - entry >= activation:
                stop = max(stop, best - gap)
        else:
            if opens[index] >= stop:
                return entry - opens[index], index, risk
            if highs[index] >= stop:
                return entry - stop, index, risk
            best = min(best, lows[index])
            if entry - best >= activation:
                stop = min(stop, best + gap)

        previous_close = closes[index]
        previous_time = times[index]

    exit_index = end - 1
    return direction * (closes[exit_index] - entry), exit_index, risk


def main():
    base.DATA_PATH = M5_PATH
    bars = base.load_bars()
    m5_times = [m1_data.parse_m5_minute(bar[0]) for bar in bars]
    m5_directions = build_m5_context(bars)
    (
        times,
        opens,
        highs,
        lows,
        closes,
        duplicate_rows,
        conflicting_timestamps,
        archive_count,
    ) = m1_data.load_m1_archives()
    atr = minute_atr(highs, lows)

    train_end = int(len(bars) * 0.70)
    validation_end = int(len(bars) * 0.85)
    split_bounds = {
        "train": (0, train_end),
        "validation": (train_end, validation_end),
        "test_diagnostic_only": (validation_end, len(bars)),
    }
    split_trades = {}
    signal_counts = {}

    for split_name, (split_start, split_end) in split_bounds.items():
        m1_start = bisect.bisect_left(times, m5_times[split_start])
        m1_end = (
            bisect.bisect_left(times, m5_times[split_end])
            if split_end < len(bars)
            else len(times)
        )
        trades = []
        active_days = {
            dt.date.fromordinal(times[index] // 1440).isoformat()
            for index in range(m1_start, m1_end)
        }
        signal_count = 0
        next_entry_index = m1_start
        for index in range(max(m1_start, 1), min(m1_end - 1, len(times) - 1)):
            if times[index] != times[index - 1] + 1:
                continue
            m5_index = bisect.bisect_right(m5_times, times[index] - 5) - 1
            if m5_index < split_start or m5_index >= split_end:
                continue
            direction = m5_directions[m5_index]
            if direction == 0:
                continue

            candle_range = max(highs[index] - lows[index], 1e-8)
            body = closes[index] - opens[index]
            if (
                direction * body <= 0.0
                or abs(body) / candle_range < MIN_BODY_TO_RANGE
                or direction * (closes[index] - closes[index - 1]) <= 0.0
            ):
                continue
            signal_count += 1

            entry_index = index + 1
            if entry_index < next_entry_index or entry_index >= m1_end:
                continue
            if times[entry_index] != times[index] + 1:
                continue
            pnl, exit_index, risk = simulate_trade(
                times,
                opens,
                highs,
                lows,
                closes,
                entry_index,
                direction,
                atr[index],
                m1_end,
            )
            trades.append(
                {
                    "R": pnl / risk,
                    "day": dt.date.fromordinal(times[entry_index] // 1440).isoformat(),
                }
            )
            next_entry_index = exit_index + 1

        signal_counts[split_name] = signal_count
        split_trades[split_name] = {
            f"cost_{cost:g}R": summarize(trades, cost, active_days)
            for cost in COST_SCENARIOS_R
        }

    payload = {
        "data": {
            "m5_bars": len(bars),
            "m1_bars": len(times),
            "m1_archives": archive_count,
            "duplicate_m1_rows_skipped": duplicate_rows,
            "conflicting_m1_timestamps_skipped": conflicting_timestamps,
        },
        "strategy": {
            "m5_context": (
                "EMA(5)/EMA(13) alignment and EMA(13) slope, with close "
                "above/below the 20-bar Bollinger midline"
            ),
            "m1_entry": (
                "M1 candle body aligned with M5 direction, "
                + (
                    f"body/range at least {MIN_BODY_TO_RANGE:.2f}, "
                    if MIN_BODY_TO_RANGE > 0.0
                    else "no minimum body/range threshold, "
                )
                + "and close beyond prior M1 close"
            ),
            "entry_price": "Next consecutive M1 bar open",
            "position_policy": "One open position at a time",
            "initial_stop_atr": STOP_ATR,
            "trailing_activation_atr": TRAIL_ACTIVATION_ATR,
            "trailing_gap_atr": TRAIL_GAP_ATR,
            "max_hold_m1_bars": MAX_HOLD_M1_BARS,
            "take_profit": "None",
        },
        "signal_counts_before_position_overlap": signal_counts,
        "splits": split_trades,
        "cost_note": (
            "Bid-only data; costs are hypothetical R deductions, not broker "
            "spread or slippage quotes."
        ),
        "split_note": (
            "Chronological 70/15/15 split. The test period is diagnostic only "
            "because earlier research inspected it."
        ),
    }
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2))
    print(f"Saved {OUTPUT_PATH}")
    for split, results in split_trades.items():
        print(
            f"{split}: signals={signal_counts[split]}, "
            f"positions={results['cost_0.02R']['positions']}, "
            f"mean/day={results['cost_0.02R']['mean_positions_per_active_day']:.1f}, "
            f"days>=100={results['cost_0.02R']['active_days_at_least_100']}/"
            f"{results['cost_0.02R']['active_day_count']}, "
            f"avgNetR@0.02={results['cost_0.02R']['average_net_R']:.4f}"
        )


if __name__ == "__main__":
    main()
