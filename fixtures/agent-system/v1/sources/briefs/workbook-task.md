# Brief: thickness assessment workbook

**Deliverable:** one Excel workbook (`.xlsx`) that recomputes when a reading
changes.

## Sheet: Readings

| Point | Thickness (mm) |
| ----- | -------------- |
| A     | 9.4            |
| B     | 8.9            |
| C     | 8.2            |
| D     | 9.1            |

## Sheet: Assessment

Four rows, each a **live formula** rather than a typed-in number:

1. `Governing` — the minimum of the four readings.
2. `Minimum allowable` — the applicable value, with the SOP revision named in
   an adjacent cell.
3. `Wall loss %` — `(minimum - governing) / minimum * 100`.
4. `Below minimum?` — a comparison that returns a word, not a number.

## What makes this pass

Changing `C` from 8.2 to 9.3 in the Readings sheet must change `Governing`,
`Wall loss %` and `Below minimum?`. A workbook whose Assessment sheet holds
typed constants produces the right numbers once and is wrong for ever after,
which is the failure this task exists to detect.
