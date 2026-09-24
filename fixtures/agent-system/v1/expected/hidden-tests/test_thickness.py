"""Independent tests for the coding fixture.

Independent in the sense that matters: the coding agent never sees this file,
so it cannot be written to satisfy these assertions specifically. It is run
against whatever `thickness.py` the agent produced, imported by path.

Run:
    python -m pytest fixtures/agent-system/v1/expected/hidden-tests/test_thickness.py \
        --submission <path to the produced thickness.py>

or, without pytest:
    python fixtures/agent-system/v1/expected/hidden-tests/test_thickness.py <path>
"""

import importlib.util
import math
import sys
from pathlib import Path


def load(path):
    spec = importlib.util.spec_from_file_location("submission_thickness", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"{path} is not importable as a Python module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CASES = []


def case(name):
    def register(fn):
        CASES.append((name, fn))
        return fn
    return register


@case("wall_loss_base")
def _(m):
    got = m.wall_loss_percent(9.0, 8.2)
    assert math.isclose(got, 8.888888888888889, rel_tol=1e-9), got


@case("wall_loss_divides_by_the_minimum_not_the_measurement")
def _(m):
    # (9.0-8.2)/8.2*100 = 9.756..., the answer a wrong denominator gives.
    got = m.wall_loss_percent(9.0, 8.2)
    assert not math.isclose(got, 9.756097560975610, rel_tol=1e-6), (
        "divided by the measurement instead of the minimum")


@case("wall_loss_at_the_minimum_is_zero")
def _(m):
    assert math.isclose(m.wall_loss_percent(9.0, 9.0), 0.0, abs_tol=1e-12)


@case("wall_loss_above_the_minimum_is_negative_not_an_error")
def _(m):
    got = m.wall_loss_percent(9.0, 9.6)
    assert got < 0, got


@case("wall_loss_rejects_a_zero_minimum")
def _(m):
    try:
        got = m.wall_loss_percent(0.0, 6.4)
    except ValueError:
        return
    raise AssertionError(f"returned {got!r} for a zero minimum instead of raising ValueError")


@case("wall_loss_rejects_a_negative_minimum")
def _(m):
    try:
        got = m.wall_loss_percent(-1.0, 6.4)
    except ValueError:
        return
    raise AssertionError(f"returned {got!r} for a negative minimum instead of raising ValueError")


@case("replacement_due_crosses_two_month_boundaries")
def _(m):
    got = m.replacement_due("2026-08-12", 90)
    assert got == "2026-11-10", got


@case("replacement_due_is_days_not_months")
def _(m):
    # 2026-11-12 is what "add three months" produces.
    got = m.replacement_due("2026-08-12", 90)
    assert got != "2026-11-12", "added three calendar months instead of 90 days"


@case("replacement_due_handles_a_leap_day")
def _(m):
    got = m.replacement_due("2028-02-28", 2)
    assert got == "2028-03-01", got


@case("replacement_due_rejects_a_negative_window")
def _(m):
    try:
        got = m.replacement_due("2026-08-12", -1)
    except ValueError:
        return
    raise AssertionError(f"returned {got!r} for a negative window instead of raising ValueError")


@case("replacement_due_rejects_an_unparseable_date")
def _(m):
    try:
        got = m.replacement_due("12 August 2026", 90)
    except ValueError:
        return
    raise AssertionError(f"returned {got!r} for an unparseable date instead of raising ValueError")


def main(argv):
    if len(argv) < 2:
        print("usage: test_thickness.py <path to thickness.py>", file=sys.stderr)
        return 2
    path = Path(argv[1])
    if not path.exists():
        print(f"FAILED  submission not found at {path}", file=sys.stderr)
        return 1
    try:
        module = load(path)
    except Exception as error:  # noqa: BLE001 - reporting, not handling
        print(f"FAILED  {path} did not import: {error}", file=sys.stderr)
        return 1

    passed, failed = 0, []
    for name, fn in CASES:
        try:
            fn(module)
        except Exception as error:  # noqa: BLE001 - reporting, not handling
            failed.append((name, error))
            print(f"FAIL  {name}: {error}")
        else:
            passed += 1
            print(f"PASS  {name}")

    print(f"\n{passed} passed, {len(failed)} failed, {len(CASES)} selected")
    # Zero selected cases is not a pass. It is a broken harness, and it exits
    # non-zero so nothing downstream reads it as coverage.
    if not CASES:
        print("FAILED  no cases were selected", file=sys.stderr)
        return 1
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
