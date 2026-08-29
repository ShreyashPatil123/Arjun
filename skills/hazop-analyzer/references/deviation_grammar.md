# HAZOP deviation grammar

This is a reference for the model, not a user-facing document.

HAZOP deviations are written with one of seven guidewords applied to a
process parameter:

| Guideword | Meaning | Example |
|---|---|---|
| No / Not | Complete absence of the parameter | No flow |
| More | Quantitative increase | More pressure |
| Less | Quantitative decrease | Less temperature |
| Reverse | Logical opposite | Reverse flow |
| As well as | Qualitative addition | As well as water (i.e. two-phase) |
| Part of | Qualitative reduction | Part of component (i.e. wrong composition) |
| Other than | Complete substitution | Other than nitrogen (i.e. inert failure) |

A deviation row is only well-formed if the **parameter** is one of
the standard HAZOP parameters: flow, pressure, temperature, level,
composition, phase, reaction, mix, residence time, sampling,
maintenance, start-up, shutdown, SIS, utilities.

The skill rejects deviations that do not match this grammar and
suggests the closest valid deviation instead.
