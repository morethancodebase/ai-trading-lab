"""Day 6: realistic Indian equity intraday trading-cost model.

A round-trip trade (entry + exit) incurs brokerage, STT (sell-side), exchange
transaction charges, SEBI turnover fee, stamp duty (buy-side), GST on the
chargeable components, and slippage (bid/ask spread + market impact) on both
legs. All components are configurable.

``ZERO_COST`` disables every component (the gross-edge baseline, matching the
Day 5 convention that returns were reported raw, before any costs).
``DEFAULT_COST`` uses Zerodha's current equity-intraday (NSE) statutory
charges only (brokerage, STT, exchange, SEBI, stamp, GST). Slippage defaults
to 0 -- it is a market-impact assumption, not a Zerodha charge; set
``slippage_bps > 0`` on a custom ``CostModel`` to add one.

Costs are computed per round-trip on the actual entry/exit leg values
(qty x price), so they scale with notional and with the small price drift
during the 5-minute hold. ``round_trip_cost`` is vectorised (numpy) so a whole
ledger can be costed in one call; ``round_trip_breakdown`` gives the per-component
rupee split for a single trade.

Only numpy is used (already in the project stack).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CostModel:
    """Round-trip cost model for Indian equity intraday trades.

    All ``*_bps`` fields are in basis points (1 bp = 0.01% = 1e-4). Monetary
    fields are in rupees. Defaults are representative FY2024-25 NSE equity
    intraday rates and are deliberately conservative.
    """

    # Brokerage per executed order, Zerodha equity intraday: Rs 20 per order OR
    # 0.03% of the order value, whichever is lower (applied to BOTH the entry
    # and exit legs). For a Rs 1,00,000 leg, 0.03% = Rs 30 > Rs 20, so the flat
    # Rs 20 applies; for very small legs (< ~Rs 66,667) the 0.03% cap is lower.
    brokerage_per_order: float = 20.0
    brokerage_pct: float = 0.0003        # 0.03% per order (the "whichever lower" cap)
    # STT (Securities Transaction Tax) on the SELL leg, in bps. Zerodha's
    # current equity-intraday STT is 0.025% (2.5 bps) on the sell side. (The
    # 0.1% rate applies to equity DELIVERY, not intraday.)
    stt_bps: float = 2.5
    # Exchange transaction charge (NSE) on turnover, in bps of (entry+exit)
    # turnover. Zerodha's current NSE equity-intraday txn charge is 0.00307%.
    exchange_bps: float = 0.307
    # SEBI turnover fee: rupees per Rs 1 crore (1e7) of turnover. Rs 10/crore.
    sebi_per_crore: float = 10.0
    # Stamp duty on the BUY leg, in bps. Equity intraday ~ 0.003% = 0.3 bps.
    stamp_bps: float = 0.3
    # GST on (brokerage + exchange + SEBI), as a fraction. 18%.
    gst_pct: float = 0.18
    # Slippage per leg, in bps (bid/ask spread + market impact). Applied to both
    # the entry and exit leg values. This is a market-impact ASSUMPTION, not a
    # Zerodha statutory charge; it defaults to 0 so DEFAULT_COST matches Zerodha's
    # fee table exactly. Set it > 0 to add a slippage / market-impact assumption.
    slippage_bps: float = 0.0

    def round_trip_cost(self, direction, entry_value, exit_value):
        """Total rupee cost of one round-trip trade (scalar or numpy arrays).

        ``direction`` is +1 (LONG: buy entry, sell exit) or -1 (SHORT: sell
        entry, buy exit). STT is charged on the sell leg and stamp duty on the
        buy leg; brokerage / exchange / SEBI / slippage are leg-symmetric.
        """
        direction = np.asarray(direction, dtype="float64")
        entry_value = np.asarray(entry_value, dtype="float64")
        exit_value = np.asarray(exit_value, dtype="float64")
        sell_value = np.where(direction == 1.0, exit_value, entry_value)
        buy_value = np.where(direction == 1.0, entry_value, exit_value)
        turnover = entry_value + exit_value
        brokerage = np.minimum(self.brokerage_per_order, self.brokerage_pct * entry_value) \
            + np.minimum(self.brokerage_per_order, self.brokerage_pct * exit_value)
        stt = self.stt_bps / 1e4 * sell_value
        exchange = self.exchange_bps / 1e4 * turnover
        sebi = self.sebi_per_crore / 1e7 * turnover
        stamp = self.stamp_bps / 1e4 * buy_value
        gst = self.gst_pct * (brokerage + exchange + sebi)
        slippage = self.slippage_bps / 1e4 * turnover
        return brokerage + stt + exchange + sebi + stamp + gst + slippage

    def round_trip_breakdown(self, direction, entry_value, exit_value) -> dict:
        """Per-component rupee cost breakdown for a single trade (scalar inputs)."""
        direction = float(np.asarray(direction))
        entry_value = float(np.asarray(entry_value))
        exit_value = float(np.asarray(exit_value))
        sell_value = exit_value if direction == 1.0 else entry_value
        buy_value = entry_value if direction == 1.0 else exit_value
        turnover = entry_value + exit_value
        brokerage = (min(self.brokerage_per_order, self.brokerage_pct * entry_value)
                     + min(self.brokerage_per_order, self.brokerage_pct * exit_value))
        stt = self.stt_bps / 1e4 * sell_value
        exchange = self.exchange_bps / 1e4 * turnover
        sebi = self.sebi_per_crore / 1e7 * turnover
        stamp = self.stamp_bps / 1e4 * buy_value
        gst = self.gst_pct * (brokerage + exchange + sebi)
        slippage = self.slippage_bps / 1e4 * turnover
        return {
            "brokerage": brokerage,
            "stt": stt,
            "exchange": exchange,
            "sebi": sebi,
            "stamp": stamp,
            "gst": gst,
            "slippage": slippage,
            "total": brokerage + stt + exchange + sebi + stamp + gst + slippage,
        }

    def params(self) -> dict:
        return {
            "brokerage_per_order": self.brokerage_per_order,
            "brokerage_pct": self.brokerage_pct,
            "stt_bps": self.stt_bps,
            "exchange_bps": self.exchange_bps,
            "sebi_per_crore": self.sebi_per_crore,
            "stamp_bps": self.stamp_bps,
            "gst_pct": self.gst_pct,
            "slippage_bps": self.slippage_bps,
        }


# Gross-edge baseline: every component disabled (returns reported raw, matching
# the Day 5 "no costs" convention).
ZERO_COST = CostModel(
    brokerage_per_order=0.0,
    brokerage_pct=0.0,
    stt_bps=0.0,
    exchange_bps=0.0,
    sebi_per_crore=0.0,
    stamp_bps=0.0,
    gst_pct=0.0,
    slippage_bps=0.0,
)

# Zerodha current equity-intraday (NSE) statutory charges only -- brokerage
# (Rs 20 or 0.03%/order, whichever lower), STT 0.025% sell-side, NSE txn
# 0.00307%, SEBI Rs 10/crore, stamp 0.003% buy-side, GST 18%. No slippage /
# market-impact assumption (slippage_bps defaults to 0).
DEFAULT_COST = CostModel()
