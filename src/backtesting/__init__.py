"""Day 6: cost-aware backtesting engine for Day 5 directional signals.

Reuses Day 5's signal logic (``src.analysis.signal_discovery``) verbatim and
adds realistic Indian equity intraday trading costs, a 5-minute next-open /
same-session execution model, and per-trade / equity / per-stock / time-of-day
metrics. See ``src.backtesting.run_backtest`` for the CLI.
"""
from src.backtesting.costs import DEFAULT_COST, ZERO_COST, CostModel
from src.backtesting.engine import (
    COST_MODELS,
    BacktestConfig,
    apply_costs,
    compute_metrics,
    equity_curve,
    load_stock,
    run_backtest,
    simulate_stock,
)

__all__ = [
    "CostModel",
    "DEFAULT_COST",
    "ZERO_COST",
    "COST_MODELS",
    "BacktestConfig",
    "apply_costs",
    "compute_metrics",
    "equity_curve",
    "load_stock",
    "run_backtest",
    "simulate_stock",
]
