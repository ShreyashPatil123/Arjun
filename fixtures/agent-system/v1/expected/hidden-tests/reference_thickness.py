"""A correct implementation, held out alongside the tests.

It exists to prove the hidden tests are satisfiable. A test suite nobody has
ever seen pass is not a grading key — it is an unfalsifiable way to fail every
submission, and it would make the coding fixture worthless in exactly the
direction that looks rigorous.

The harness runs the tests against this file and requires PASSED, and against
`../../sources/code/failing-example.py` and requires FAILED. Both must hold or
the fixture pack is reported as broken.
"""

from datetime import date, timedelta


def wall_loss_percent(minimum_mm: float, measured_mm: float) -> float:
    if minimum_mm <= 0:
        raise ValueError(
            f"minimum allowable thickness must be positive; got {minimum_mm!r}")
    return (minimum_mm - measured_mm) / minimum_mm * 100


def replacement_due(inspection_date: str, window_days: int) -> str:
    if window_days < 0:
        raise ValueError(f"the replacement window cannot be negative; got {window_days!r}")
    try:
        start = date.fromisoformat(inspection_date)
    except ValueError as error:
        raise ValueError(
            f"inspection_date must be YYYY-MM-DD; got {inspection_date!r}") from error
    return (start + timedelta(days=window_days)).isoformat()
