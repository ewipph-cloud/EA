import array
import bisect
import csv
import datetime as dt
import io
import json
import zipfile
from pathlib import Path

import research_trendline_setups as setups
import train_swing_model as base


ROOT = Path(__file__).parent
DATA_PATH = ROOT / "XAUUSD_M5_HistData_2022-2026_clean.csv"
OUTPUT_PATH = ROOT / "xauusd_m5_m1_trailing_execution.json"
SWING_SCALE = 0.75
SETUP = "support_rejection_uptrend"
DIRECTION = 1
MAX_M1_BARS = 60
CONFIG = {
    "initial_sl": 0.75,
    "trail_activation": 1.0,
    "trail_gap": 0.75,
}
COST_SCENARIOS_R = (0.0, 0.02, 0.05, 0.10)


def minute_number(year, month, day, hour, minute):
    ordinal = dt.date(year, month, day).toordinal()
    return ordinal * 1440 + hour * 60 + minute


def parse_m5_minute(timestamp):
    parsed = dt.datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
    return minute_number(
        parsed.year, parsed.month, parsed.day, parsed.hour, parsed.minute
    )


def load_m1_archives():
    times = array.array("q")
    opens = array.array("d")
    highs = array.array("d")
    lows = array.array("d")
    closes = array.array("d")
    duplicate_rows = 0
    archive_paths = sorted(ROOT.glob("HISTDATA_XAUUSD_M1_*.zip"))
    if not archive_paths:
        raise FileNotFoundError("No locally archived HistData XAUUSD M1 files")

    for path in archive_paths:
        with zipfile.ZipFile(path) as archive:
            csv_name = next(
                (name for name in archive.namelist() if name.endswith(".csv")),
                None,
            )
            if csv_name is None:
                continue
            with archive.open(csv_name) as raw:
                text = io.TextIOWrapper(raw, encoding="utf-8")
                for row in csv.reader(text, delimiter=";"):
                    if len(row) < 5:
                        continue
                    stamp = row[0]
                    day, clock = stamp.split()
                    current_time = minute_number(
                        int(day[:4]),
                        int(day[4:6]),
                        int(day[6:8]),
                        int(clock[:2]),
                        int(clock[2:4]),
                    )
                    times.append(current_time)
                    opens.append(float(row[1]))
                    highs.append(float(row[2]))
                    lows.append(float(row[3]))
                    closes.append(float(row[4]))

    order = sorted(range(len(times)), key=times.__getitem__)
    sorted_times = array.array("q")
    sorted_opens = array.array("d")
    sorted_highs = array.array("d")
    sorted_lows = array.array("d")
    sorted_closes = array.array("d")
    conflicting_timestamps = 0
    position = 0
    while position < len(order):
        end = position + 1
        timestamp = times[order[position]]
        while end < len(order) and times[order[end]] == timestamp:
            end += 1
        group = order[position:end]
        price_rows = {
            (opens[index], highs[index], lows[index], closes[index])
            for index in group
        }
        duplicate_rows += len(group) - 1
        if len(price_rows) > 1:
            conflicting_timestamps += 1
            position = end
            continue
        sorted_times.append(timestamp)
        open_price, high_price, low_price, close_price = next(iter(price_rows))
        sorted_opens.append(open_price)
        sorted_highs.append(high_price)
        sorted_lows.append(low_price)
        sorted_closes.append(close_price)
        position = end
    return (
        sorted_times,
        sorted_opens,
        sorted_highs,
        sorted_lows,
        sorted_closes,
        duplicate_rows,
        conflicting_timestamps,
        len(archive_paths),
    )


def simulate_m1_trade(
    times,
    opens,
    highs,
    lows,
    closes,
    m1_start,
    entry_price,
    scale,
    direction,
    split_end,
    config=None,
    max_m1_bars=None,
):
    config = config or CONFIG
    max_m1_bars = max_m1_bars or MAX_M1_BARS
    risk = config["initial_sl"] * max(scale, 1e-8)
    stop = entry_price - direction * risk
    best_price = entry_price
    activation = config["trail_activation"] * max(scale, 1e-8)
    trail_gap = config["trail_gap"] * max(scale, 1e-8)
    end_index = min(m1_start + max_m1_bars, split_end)
    previous_close = entry_price
    previous_time = times[m1_start - 1] if m1_start else times[m1_start] - 1

    for index in range(m1_start, end_index):
        if times[index] - previous_time > 1:
            pnl = direction * (previous_close - entry_price)
            return pnl, index - 1, "session_gap"

        open_price = opens[index]
        high = highs[index]
        low = lows[index]
        if direction == 1:
            if open_price <= stop:
                return open_price - entry_price, index, "stop"
            if low <= stop:
                return stop - entry_price, index, "stop"
            best_price = max(best_price, high)
            if best_price - entry_price >= activation:
                stop = max(stop, best_price - trail_gap)
        else:
            if open_price >= stop:
                return entry_price - open_price, index, "stop"
            if high >= stop:
                return entry_price - stop, index, "stop"
            best_price = min(best_price, low)
            if entry_price - best_price >= activation:
                stop = min(stop, best_price + trail_gap)

        previous_close = closes[index]
        previous_time = times[index]

    exit_index = end_index - 1
    return direction * (closes[exit_index] - entry_price), exit_index, "time"


def summarize(trades, cost):
    if not trades:
        return {"trades": 0, "average_net_R": None, "total_net_R": 0.0}
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
        "stop_exit_rate": sum(
            trade["exit_reason"] == "stop" for trade in trades
        )
        / len(trades),
        "session_gap_exit_rate": sum(
            trade["exit_reason"] == "session_gap" for trade in trades
        )
        / len(trades),
    }


def main():
    setups.DATA_PATH = DATA_PATH
    base.DATA_PATH = DATA_PATH
    base.SWING_SCALE = SWING_SCALE
    setups.SWING_SCALE = SWING_SCALE
    setups.events = {}
    base.make_pattern_features = setups.collect_setups
    bars = base.load_bars()
    m5_scales = base.calculate_scales(bars)
    base.first_barrier_outcome = (
        lambda _bars, _scales, start_index: (0, start_index + 1)
    )
    samples, _ = base.detect_samples(bars)
    (
        times,
        opens,
        highs,
        lows,
        closes,
        duplicates,
        conflicting_timestamps,
        archive_count,
    ) = (
        load_m1_archives()
    )

    m5_times = [parse_m5_minute(bar[0]) for bar in bars]
    train_end = int(len(bars) * 0.70)
    validation_end = int(len(bars) * 0.85)
    split_bounds = {
        "train": (0, train_end),
        "validation": (train_end, validation_end),
        "test_diagnostic_only": (validation_end, len(bars)),
    }

    split_trades = {}
    skipped_no_exact_m1_bar = {}
    for split_name, (start, end) in split_bounds.items():
        if end < len(bars):
            m1_split_end = bisect.bisect_left(times, m5_times[end])
        else:
            m1_split_end = len(times)
        candidates = []
        skipped = 0
        for sample in samples:
            event_index = sample["event_index"]
            if not start <= event_index < end:
                continue
            event = setups.events.get(event_index)
            if not event or not event["rules"].get(SETUP, False):
                continue
            entry_minute = m5_times[event_index] + 5
            m1_start = bisect.bisect_left(times, entry_minute)
            if (
                m1_start >= len(times)
                or times[m1_start] != entry_minute
                or m1_start + MAX_M1_BARS >= m1_split_end
            ):
                skipped += 1
                continue
            candidates.append(
                {
                    "event_index": event_index,
                    "m1_start": m1_start,
                    "entry_price": bars[event_index][4],
                    "scale": m5_scales[event_index],
                }
            )

        trades = []
        next_entry_index = 0
        for candidate in candidates:
            if candidate["m1_start"] < next_entry_index:
                continue
            pnl, exit_index, reason = simulate_m1_trade(
                times,
                opens,
                highs,
                lows,
                closes,
                candidate["m1_start"],
                candidate["entry_price"],
                candidate["scale"],
                DIRECTION,
                m1_split_end,
            )
            risk = CONFIG["initial_sl"] * max(candidate["scale"], 1e-8)
            trades.append(
                {
                    "R": pnl / risk,
                    "exit_index": exit_index,
                    "exit_reason": reason,
                }
            )
            next_entry_index = exit_index + 1
        split_trades[split_name] = {
            "metrics": {
                f"cost_{cost:g}R": summarize(trades, cost)
                for cost in COST_SCENARIOS_R
            },
            "potential_entries": len(candidates),
            "skipped_missing_m1_or_split_boundary": skipped,
        }

    payload = {
        "source": "Existing local HistData XAUUSD M1 archives",
        "m1_archive_files": archive_count,
        "m1_bar_count": len(times),
        "duplicate_m1_rows_skipped": duplicates,
        "conflicting_m1_timestamps_skipped": conflicting_timestamps,
        "m5_signal_bars": len(bars),
        "date_range_m1_minutes": {
            "first": times[0],
            "last": times[-1],
        },
        "strategy": {
            "setup": SETUP,
            "swing_scale": SWING_SCALE,
            "direction": "long",
            "entry": "M5 signal-bar close; first following M1 bar",
            "initial_sl": CONFIG["initial_sl"],
            "trailing_activation": CONFIG["trail_activation"],
            "trailing_gap": CONFIG["trail_gap"],
            "max_hold_m1_bars": MAX_M1_BARS,
            "take_profit": "None.",
        },
        "splits": split_trades,
        "execution_notes": {
            "intrabar": (
                "Trailing updates are checked on M1. Existing stop is checked "
                "before updating from that candle's high/low."
            ),
            "market_gaps": (
                "Open positions exit at the prior M1 close when a time gap "
                "occurs, to avoid carrying a scalp across a closed session."
            ),
            "costs": (
                "Bid-only data; actual spread, commission, swap, and slippage "
                "are unavailable. Cost deductions are sensitivity scenarios."
            ),
            "test_period": (
                "Diagnostic only; prior research has inspected this period."
            ),
        },
    }
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2))
    print(f"Saved {OUTPUT_PATH}")
    print(f"M1 bars: {len(times)}, duplicate rows skipped: {duplicates}")
    for split, result in split_trades.items():
        print(f"\n{split}: potential entries={result['potential_entries']}")
        for cost, metrics in result["metrics"].items():
            print(
                f"  {cost}: n={metrics['trades']} "
                f"avgNetR={metrics['average_net_R']} "
                f"PF={metrics['net_profit_factor']}"
            )


if __name__ == "__main__":
    main()
