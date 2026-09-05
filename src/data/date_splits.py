"""Reusable train/validation/test date-split utilities (Day 3+).

Classify timestamps by calendar date. Rows outside the defined ranges return None.
No rows are removed or duplicated — this is metadata for downstream filtering.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

SplitName = Literal["train", "validation", "test"]

TRAIN_START = date(2023, 9, 4)
TRAIN_END = date(2025, 9, 3)

VALIDATION_START = date(2025, 9, 4)
VALIDATION_END = date(2026, 3, 3)

TEST_START = date(2026, 3, 4)
TEST_END = date(2026, 9, 4)

SPLIT_RANGES: dict[SplitName, tuple[date, date]] = {
    "train": (TRAIN_START, TRAIN_END),
    "validation": (VALIDATION_START, VALIDATION_END),
    "test": (TEST_START, TEST_END),
}


def _as_date(ts: date | datetime | str) -> date:
    if isinstance(ts, date) and not isinstance(ts, datetime):
        return ts
    if isinstance(ts, datetime):
        return ts.date()
    return date.fromisoformat(str(ts)[:10])


def classify_timestamp(ts: date | datetime | str) -> SplitName | None:
    """Return 'train', 'validation', 'test', or None if outside all ranges."""
    d = _as_date(ts)
    if TRAIN_START <= d <= TRAIN_END:
        return "train"
    if VALIDATION_START <= d <= VALIDATION_END:
        return "validation"
    if TEST_START <= d <= TEST_END:
        return "test"
    return None


def split_bounds() -> dict[str, dict[str, date]]:
    """Human-readable inclusive date boundaries for each split."""
    return {
        name: {"start": start, "end": end}
        for name, (start, end) in SPLIT_RANGES.items()
    }
