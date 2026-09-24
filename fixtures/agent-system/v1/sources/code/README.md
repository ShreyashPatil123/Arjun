# Coding task: thickness assessment helpers

Write `thickness.py` exposing exactly these two functions. The tests that grade
this are **not in this directory** — see `../../expected/hidden-tests/`, which
is outside the run workspace the coding agent may read or write.

## `wall_loss_percent(minimum_mm: float, measured_mm: float) -> float`

Wall loss as a percentage of the minimum allowable thickness:

    (minimum - measured) / minimum * 100

- Returns a float. Do not round; the caller decides presentation.
- A measurement at or above the minimum yields zero or a negative number, and
  that is the answer, not an error.
- `minimum_mm <= 0` has no answer. Raise `ValueError` with a message naming the
  offending value. Returning `0.0`, `float('inf')` or `None` is wrong: each
  would flow into an approval note as a figure somebody signs.

## `replacement_due(inspection_date: str, window_days: int) -> str`

The last date a vessel may remain in service.

- `inspection_date` is `YYYY-MM-DD`. The return value is `YYYY-MM-DD`.
- Add `window_days` calendar days. Not months, not "about three months".
- A negative `window_days` raises `ValueError`.
- An unparseable date raises `ValueError`, not a silent passthrough.

## Allowed

The Python standard library only. No network. No file access.

## Example (not a test)

    wall_loss_percent(9.0, 8.2)          -> 8.888888888888889
    replacement_due("2026-08-12", 90)    -> "2026-11-10"
