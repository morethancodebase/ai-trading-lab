"""Day 6 CLI: run the cost-aware backtest of Day 5 directional signals.

Examples (run from the project root)::

    python -m src.backtesting.run_backtest
    python -m src.backtesting.run_backtest --signals C1 --splits validation
    python -m src.backtesting.run_backtest --limit-stocks 5 --signals C1   # smoke

Outputs are written to reports/day6_backtest/ (local-only, gitignored). TEST is
never read; the engine asserts no row dated on/after TEST_START (2026-03-04).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.backtesting.engine import (
    COST_MODELS,
    REPORTS_DIR,
    ROOT,
    BacktestConfig,
    run_backtest,
)

DAY5_SUMMARY = ROOT / "reports" / "day5_signal_discovery" / "run_summary.json"
DAY5_TRAIN_VAL = ROOT / "reports" / "day5_signal_discovery" / "signal_train_vs_val.csv"

# Fallback "promising" set (persistence_flag=True in Day 5); the actual default
# is read from Day 5's run_summary.json when present.
PROMISING_FALLBACK = ["C1", "C3", "C2", "C4", "S1", "S1b", "S2", "S2b", "S3", "S4", "S5"]


def default_signals() -> list[str]:
    """Day 5 promising signal ids (persistence_flag=True), from run_summary.json."""
    if DAY5_SUMMARY.exists():
        try:
            with DAY5_SUMMARY.open() as f:
                s = json.load(f)
            ids = [x["signal_id"] for x in s.get("promising_signals", [])]
            if ids:
                return ids
        except Exception:
            pass
    return list(PROMISING_FALLBACK)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--signals", default=None,
                   help="Comma-separated signal ids (default: Day 5 promising set).")
    p.add_argument("--splits", default="train,validation",
                   help="Comma-separated splits from {train, validation}.")
    p.add_argument("--cost-modes", default="zero,model",
                   help=f"Comma-separated cost modes from {sorted(COST_MODELS)}.")
    p.add_argument("--notional", type=float, default=100000.0,
                   help="Fixed Rs notional per trade (default 100000).")
    p.add_argument("--initial-capital", type=float, default=1000000.0,
                   help="Equity-curve baseline in Rs (default 1000000).")
    p.add_argument("--limit-stocks", type=int, default=None,
                   help="Process only the first N stocks (smoke test).")
    p.add_argument("--write-trades", dest="write_trades", action="store_true",
                   default=True, help="Write per-trade ledger CSVs (default on).")
    p.add_argument("--no-write-trades", dest="write_trades", action="store_false",
                   help="Skip per-trade ledger CSVs.")
    p.add_argument("--max-trades-rows", type=int, default=50000,
                   help="Cap rows written per trade ledger CSV (metrics use all).")
    p.add_argument("--plot", dest="plot", action="store_true", default=True,
                   help="Render equity/drawdown PNGs (default on; needs matplotlib).")
    p.add_argument("--no-plot", dest="plot", action="store_false",
                   help="Skip PNG rendering.")
    p.add_argument("--reports-dir", default=str(REPORTS_DIR),
                   help="Output directory (default reports/day6_backtest).")
    args = p.parse_args(argv)

    def _split_csv(s, allowed, name):
        vals = [v.strip() for v in s.split(",") if v.strip()]
        bad = [v for v in vals if v not in allowed]
        if bad or not vals:
            p.error(f"invalid {name}: {bad or 'empty'}; allowed: {sorted(allowed)}")
        return vals

    if args.signals:
        signals = [s.strip() for s in args.signals.split(",") if s.strip()]
    else:
        signals = default_signals()
    if not signals:
        p.error("no signals selected")
    splits = _split_csv(args.splits, {"train", "validation"}, "splits")
    cost_modes = _split_csv(args.cost_modes, set(COST_MODELS), "cost-modes")

    config = BacktestConfig(
        signal_ids=signals,
        splits=splits,
        cost_modes=cost_modes,
        notional=args.notional,
        initial_capital=args.initial_capital,
        limit_stocks=args.limit_stocks,
        write_trades=args.write_trades,
        max_trades_rows=args.max_trades_rows,
        plot=args.plot,
    )
    run_backtest(config, Path(args.reports_dir), DAY5_TRAIN_VAL)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
