"""A deliberately wrong implementation, for the harness's honesty check.

Plan §10 journey 5 requires that a program which does not work is reported as
failed rather than described as successful. This file is that program: it is
the same shape as a correct answer and it is wrong in three ways the hidden
tests each catch.

It is a *fixture input*, never a starting point for the coding agent. The
harness runs it against the hidden tests and requires the result to be FAILED.
A harness run in which this file passes is a broken harness.
"""


def wall_loss_percent(minimum_mm: float, measured_mm: float) -> float:
    # Wrong basis: divides by the measurement instead of the minimum.
    return (minimum_mm - measured_mm) / measured_mm * 100


def replacement_due(inspection_date: str, window_days: int) -> str:
    # Wrong arithmetic: treats 90 days as three calendar months.
    year, month, day = (int(part) for part in inspection_date.split("-"))
    month += window_days // 30
    while month > 12:
        month -= 12
        year += 1
    return f"{year:04d}-{month:02d}-{day:02d}"
