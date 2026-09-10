"""Day 6: realistic Indian equity intraday trading-cost model.

A round-trip trade (entry + exit) incurs brokerage, STT (sell-side), exchange
transaction charges, SEBI turnover fee, stamp duty (buy-side), GST on the
chargeable components, and slippage (bid/ask spread + market impact) on both
legs. All components are configurable.

``ZERO_COST`` disables every component (the gross-edge baseline, matching the
Day 5 convention that returns were reported raw, before any costs).
``DEFAULT_COST`` uses representative FY2024-25 Indian equity intraday rates
(conservative; see the per-field notes, especially the STT caveat).

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

    # Brokerage: flat rupees per executed order (entry and exit are two
    # orders). Discount brokers (e.g. Zerodha) charge Rs 20/order or 0.03% of
    # turnover, whichever is lower; for a Rs 1,00,000 notional 0.03% = Rs 30,
    # so the flat Rs 20 applies. We model the flat per-order fee.
    brokerage_per_order: float = 20.0
    # STT (Securities Transaction Tax) on the SELL leg, in bps. Equity intraday
    # sell was 0.025% (2.5 bps) historically and was raised to 0.1% (10 bps)
    # effective 2024-10-01 (Budget 2024). Our 2023-2026 sample straddles that
    # change; we use the higher post-2024 rate as a conservative constant.
    stt_bps: float = 10.0
    # Exchange transaction charge (NSE) on turnover (both legs), bps per leg.
    # Equity intraday ~ 0.00325% = 0.325 bps per leg.
    exchange_bps: float = 0.325
    # SEBI turnover fee: rupees per Rs 1 crore (1e7) of turnover.
    sebi_per_crore: float = 10.0
    # Stamp duty on the BUY leg, in bps. Equity intraday ~ 0.003% = 0.3 bps.
    stamp_bps: float = 0.3
    # GST on (brokerage + exchange + SEBI), as a fraction. 18%.
    gst_pct: float = 0.18
    # Slippage per leg, in bps (bid/ask spread + market impact). Applied to both
    # the entry and exit leg values.
    slippage_bps: float = 1.0

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
        brokerage = 2.0 * self.brokerage_per_order
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
        brokerage = 2.0 * self.brokerage_per_order
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
    stt_bps=0.0,
    exchange_bps=0.0,
    sebi_per_crore=0.0,
    stamp_bps=0.0,
    gst_pct=0.0,
    slippage_bps=0.0,
)

# Representative FY2024-25 Indian equity intraday rates (conservative).
DEFAULT_COST = CostModel()
