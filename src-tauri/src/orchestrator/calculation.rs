//! Arithmetic that is actually correct, with the units checked.
//!
//! ARJUN design rule 27: *"the orchestrator should use a deterministic calculator ... The
//! model may explain the result, but the calculation engine should be the source
//! of numerical truth."*
//!
//! That instruction is not caution for its own sake. A language model asked for
//! `(8.2 - 9.0) / 9.0 * 100` will usually produce something close to `-8.9`, and
//! *close* is the problem: a wall-thickness deviation that is wrong in the second
//! decimal place still reads as an answer, gets copied into an approval note, and
//! nothing downstream can tell. So the number comes from here, and the model's
//! job is to say what it means.
//!
//! ## Units are checked, not carried along
//!
//! `8.2 mm - 9.0 mm` is a length. `8.2 mm - 9.0 kg` is a mistake, and it is
//! caught rather than silently producing `-0.8` of nothing. This matters more in
//! a refinery than almost anywhere: the arithmetic in an inspection report is
//! simple, and the errors that survive review are the dimensional ones.
//!
//! ## Everything is on the record
//!
//! A [`CalculationRecord`] carries the expression, the parsed inputs, every
//! intermediate step, the result with its unit, and the rounding applied. PS
//! step 27 asks for exactly that, and it is what lets a reviewer check a number
//! without redoing it — and what the artifact validator later reconciles the
//! document against.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use serde::{Deserialize, Serialize};

/// A quantity: a number and its dimensions.
///
/// Dimensions are exponents over base unit names — `mm` is `{mm: 1}`, an area
/// is `{mm: 2}`, a ratio is empty. Keeping the author's own unit names rather
/// than converting to SI is deliberate: an engineer who wrote `mm` wants to read
/// `mm` back, and a converted answer invites a second conversion error.
#[derive(Debug, Clone, PartialEq)]
pub struct Quantity {
    pub value: f64,
    pub units: BTreeMap<String, i32>,
}

impl Quantity {
    fn scalar(value: f64) -> Self {
        Self {
            value,
            units: BTreeMap::new(),
        }
    }

    fn with_unit(value: f64, unit: &str) -> Self {
        let mut units = BTreeMap::new();
        units.insert(unit.to_string(), 1);
        Self { value, units }
    }

    pub fn is_dimensionless(&self) -> bool {
        self.units.is_empty()
    }

    /// How the unit reads: `mm`, `mm²`, `kg/mm`, or empty for a plain number.
    pub fn unit_label(&self) -> String {
        if self.units.is_empty() {
            return String::new();
        }

        let mut numerator = Vec::new();
        let mut denominator = Vec::new();

        for (unit, power) in &self.units {
            let magnitude = power.abs();
            let text = match magnitude {
                1 => unit.clone(),
                2 => format!("{unit}²"),
                3 => format!("{unit}³"),
                n => format!("{unit}^{n}"),
            };
            if *power > 0 {
                numerator.push(text);
            } else {
                denominator.push(text);
            }
        }

        let top = if numerator.is_empty() {
            "1".to_string()
        } else {
            numerator.join("·")
        };

        if denominator.is_empty() {
            top
        } else {
            format!("{top}/{}", denominator.join("·"))
        }
    }

    fn combine(&self, other: &Self, sign: i32) -> BTreeMap<String, i32> {
        let mut units = self.units.clone();
        for (unit, power) in &other.units {
            let entry = units.entry(unit.clone()).or_insert(0);
            *entry += power * sign;
            if *entry == 0 {
                units.remove(unit);
            }
        }
        units
    }
}

/// One step of the working, so a reviewer can follow it without redoing it.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Step {
    pub description: String,
    pub result: String,
}

/// Everything ARJUN design rule 27 asks a calculation to record.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CalculationRecord {
    /// Exactly what was asked, unmodified.
    pub expression: String,
    /// The quantities read out of it, with their units.
    pub inputs: Vec<String>,
    pub steps: Vec<Step>,
    /// The result as a number, before any rounding for display.
    pub value: f64,
    pub unit: String,
    /// The result as it should be written down.
    pub formatted: String,
    /// How `formatted` was produced from `value`.
    pub rounding: String,
    /// True when the engine computed this rather than a model.
    ///
    /// Always true here. It exists so a record's provenance is on the record
    /// itself, rather than being something a reader has to know.
    pub deterministic: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CalculationError {
    pub message: String,
    /// Where in the expression, when known.
    pub position: Option<usize>,
}

impl CalculationError {
    fn at(position: usize, message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            position: Some(position),
        }
    }

    fn plain(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            position: None,
        }
    }
}

/// Significant figures kept when writing the result down.
///
/// Four is enough for the tolerances in an inspection report and short enough
/// that a reader is not misled into thinking the input was that precise.
const SIGNIFICANT_FIGURES: usize = 4;

struct Parser<'a> {
    input: &'a [u8],
    position: usize,
    steps: Vec<Step>,
    inputs: Vec<String>,
}

impl<'a> Parser<'a> {
    fn new(input: &'a str) -> Self {
        Self {
            input: input.as_bytes(),
            position: 0,
            steps: Vec::new(),
            inputs: Vec::new(),
        }
    }

    fn skip_spaces(&mut self) {
        while self.position < self.input.len() && self.input[self.position].is_ascii_whitespace() {
            self.position += 1;
        }
    }

    fn peek(&mut self) -> Option<u8> {
        self.skip_spaces();
        self.input.get(self.position).copied()
    }

    /// Sum: term (('+' | '-') term)*
    fn expression(&mut self) -> Result<Quantity, CalculationError> {
        let mut left = self.term()?;

        while let Some(op) = self.peek() {
            if op != b'+' && op != b'-' {
                break;
            }
            let at = self.position;
            self.position += 1;
            let right = self.term()?;

            // The check that catches the errors which survive review.
            if left.units != right.units {
                return Err(CalculationError::at(
                    at,
                    format!(
                        "Cannot {} {} and {}: the units do not match.",
                        if op == b'+' { "add" } else { "subtract" },
                        describe_unit(&left),
                        describe_unit(&right)
                    ),
                ));
            }

            let value = if op == b'+' {
                left.value + right.value
            } else {
                left.value - right.value
            };

            self.record_step(&left, op as char, &right, value, &left.units);
            left = Quantity { value, units: left.units };
        }

        Ok(left)
    }

    /// Product: factor (('*' | '/') factor)*
    fn term(&mut self) -> Result<Quantity, CalculationError> {
        let mut left = self.factor()?;

        while let Some(op) = self.peek() {
            if op != b'*' && op != b'/' {
                break;
            }
            let at = self.position;
            self.position += 1;
            let right = self.factor()?;

            if op == b'/' && right.value == 0.0 {
                return Err(CalculationError::at(at, "Division by zero."));
            }

            let value = if op == b'*' {
                left.value * right.value
            } else {
                left.value / right.value
            };
            let units = left.combine(&right, if op == b'*' { 1 } else { -1 });

            self.record_step(&left, op as char, &right, value, &units);
            left = Quantity { value, units };
        }

        Ok(left)
    }

    /// A number with an optional unit, a parenthesised expression, or a negation.
    fn factor(&mut self) -> Result<Quantity, CalculationError> {
        match self.peek() {
            Some(b'(') => {
                self.position += 1;
                let inner = self.expression()?;
                match self.peek() {
                    Some(b')') => {
                        self.position += 1;
                        Ok(inner)
                    }
                    _ => Err(CalculationError::at(
                        self.position,
                        "A bracket was opened and never closed.",
                    )),
                }
            }
            Some(b'-') => {
                self.position += 1;
                let inner = self.factor()?;
                Ok(Quantity {
                    value: -inner.value,
                    units: inner.units,
                })
            }
            Some(c) if c.is_ascii_digit() || c == b'.' => self.number(),
            Some(c) => Err(CalculationError::at(
                self.position,
                format!("Did not expect {:?} here.", c as char),
            )),
            None => Err(CalculationError::plain("The expression ended unexpectedly.")),
        }
    }

    fn number(&mut self) -> Result<Quantity, CalculationError> {
        self.skip_spaces();
        let start = self.position;

        while self.position < self.input.len()
            && (self.input[self.position].is_ascii_digit() || self.input[self.position] == b'.')
        {
            self.position += 1;
        }

        let text = std::str::from_utf8(&self.input[start..self.position]).unwrap_or("");
        let value: f64 = text
            .parse()
            .map_err(|_| CalculationError::at(start, format!("{text:?} is not a number.")))?;

        // A unit is letters or a percent sign immediately following, so
        // `8.2 mm` reads as a length and `8.2 * mm` would not silently do so.
        self.skip_spaces();
        let unit_start = self.position;
        while self.position < self.input.len()
            && (self.input[self.position].is_ascii_alphabetic()
                || self.input[self.position] == b'%')
        {
            self.position += 1;
        }

        let quantity = if unit_start == self.position {
            Quantity::scalar(value)
        } else {
            let unit = std::str::from_utf8(&self.input[unit_start..self.position]).unwrap_or("");

            // A unit is followed by an operator, a bracket, or the end. A second
            // bare word after it means this is prose that happens to contain a
            // number — "2 and then some" — not "8.2 mm". Refusing here gives a
            // message about the sentence rather than a confusing one about
            // mismatched units, and either way no number is produced.
            let after = {
                let mut probe = self.position;
                while probe < self.input.len() && self.input[probe].is_ascii_whitespace() {
                    probe += 1;
                }
                self.input.get(probe).copied()
            };
            if after.is_some_and(|c| c.is_ascii_alphabetic()) {
                return Err(CalculationError::at(
                    unit_start,
                    format!(
                        "{unit:?} does not read as a unit here. A calculation should be an                          expression such as `(8.2 mm - 9.0 mm) / 9.0 mm * 100`, not a sentence."
                    ),
                ));
            }

            Quantity::with_unit(value, unit)
        };

        self.inputs.push(format_quantity(&quantity));
        Ok(quantity)
    }

    fn record_step(
        &mut self,
        left: &Quantity,
        op: char,
        right: &Quantity,
        value: f64,
        units: &BTreeMap<String, i32>,
    ) {
        let result = Quantity {
            value,
            units: units.clone(),
        };
        self.steps.push(Step {
            description: format!(
                "{} {op} {}",
                format_quantity(left),
                format_quantity(right)
            ),
            result: format_quantity(&result),
        });
    }
}

fn describe_unit(quantity: &Quantity) -> String {
    if quantity.is_dimensionless() {
        "a plain number".to_string()
    } else {
        format!("a value in {}", quantity.unit_label())
    }
}

fn format_quantity(quantity: &Quantity) -> String {
    let label = quantity.unit_label();
    if label.is_empty() {
        trim_number(quantity.value)
    } else {
        format!("{} {label}", trim_number(quantity.value))
    }
}

/// Writes a number without trailing zeros or floating-point noise.
fn trim_number(value: f64) -> String {
    if !value.is_finite() {
        return value.to_string();
    }
    let mut text = format!("{value:.*}", 10);
    while text.contains('.') && (text.ends_with('0') || text.ends_with('.')) {
        text.pop();
    }
    if text.is_empty() || text == "-" {
        text.push('0');
    }
    text
}

/// Rounds to significant figures for display, leaving `value` exact.
fn round_significant(value: f64, figures: usize) -> f64 {
    if value == 0.0 || !value.is_finite() {
        return value;
    }
    let magnitude = value.abs().log10().floor() as i32;
    let factor = 10f64.powi(figures as i32 - 1 - magnitude);
    (value * factor).round() / factor
}

/// Evaluates an expression and records how it got there.
///
/// Never calls a model, and never will. The whole value of this function is
/// that its answer does not depend on one.
pub fn evaluate(expression: &str) -> Result<CalculationRecord, CalculationError> {
    let trimmed = expression.trim();
    if trimmed.is_empty() {
        return Err(CalculationError::plain("There is nothing to calculate."));
    }

    let mut parser = Parser::new(trimmed);
    let result = parser.expression()?;

    // Anything left over means the expression was not fully understood, and a
    // partial reading is far worse than a refusal — it produces a number.
    parser.skip_spaces();
    if parser.position < parser.input.len() {
        let rest = std::str::from_utf8(&parser.input[parser.position..]).unwrap_or("");
        return Err(CalculationError::at(
            parser.position,
            format!("Could not make sense of {rest:?}."),
        ));
    }

    if !result.value.is_finite() {
        return Err(CalculationError::plain(
            "The result is not a finite number. Check for a division by something very small.",
        ));
    }

    let rounded = round_significant(result.value, SIGNIFICANT_FIGURES);
    let unit = result.unit_label();

    let mut formatted = trim_number(rounded);
    if !unit.is_empty() {
        let _ = write!(formatted, " {unit}");
    }

    Ok(CalculationRecord {
        expression: trimmed.to_string(),
        inputs: parser.inputs,
        steps: parser.steps,
        value: result.value,
        unit,
        formatted,
        rounding: format!("{SIGNIFICANT_FIGURES} significant figures"),
        deterministic: true,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn value_of(expression: &str) -> f64 {
        evaluate(expression).expect("should evaluate").value
    }

    #[test]
    fn plain_arithmetic_is_correct() {
        assert_eq!(value_of("2 + 3"), 5.0);
        assert_eq!(value_of("10 - 4"), 6.0);
        assert_eq!(value_of("6 * 7"), 42.0);
        assert_eq!(value_of("9 / 3"), 3.0);
    }

    #[test]
    fn multiplication_binds_tighter_than_addition() {
        assert_eq!(value_of("2 + 3 * 4"), 14.0);
        assert_eq!(value_of("(2 + 3) * 4"), 20.0);
    }

    #[test]
    fn negation_works_at_the_front_and_inside_brackets() {
        assert_eq!(value_of("-5 + 2"), -3.0);
        assert_eq!(value_of("3 * (-2)"), -6.0);
    }

    /// The calculation from the problem statement's own worked example.
    #[test]
    fn the_wall_thickness_deviation_is_exact() {
        let record = evaluate("(8.2 mm - 9.0 mm) / 9.0 mm * 100").unwrap();

        assert!((record.value - -8.888_888_888_9).abs() < 1e-9);
        assert_eq!(record.formatted, "-8.889");
        // The units cancel, so the answer is a plain ratio rather than "mm".
        assert!(record.unit.is_empty());
    }

    #[test]
    fn a_length_minus_a_length_is_a_length() {
        let record = evaluate("9.0 mm - 8.2 mm").unwrap();
        assert_eq!(record.unit, "mm");
        assert_eq!(record.formatted, "0.8 mm");
    }

    /// The error class that survives human review.
    #[test]
    fn adding_incompatible_units_is_refused_rather_than_computed() {
        let error = evaluate("8.2 mm + 9.0 kg").unwrap_err();
        assert!(error.message.contains("units do not match"), "{}", error.message);
        assert!(error.message.contains("mm"));
        assert!(error.message.contains("kg"));
    }

    #[test]
    fn subtracting_a_bare_number_from_a_measurement_is_refused() {
        let error = evaluate("8.2 mm - 9.0").unwrap_err();
        assert!(error.message.contains("units do not match"));
        assert!(error.message.contains("a plain number"));
    }

    #[test]
    fn multiplying_units_produces_a_compound_unit() {
        assert_eq!(evaluate("3 mm * 4 mm").unwrap().unit, "mm²");
        assert_eq!(evaluate("10 kg / 2 mm").unwrap().unit, "kg/mm");
    }

    #[test]
    fn dividing_like_units_cancels_them() {
        let record = evaluate("8.2 mm / 9.0 mm").unwrap();
        assert!(record.unit.is_empty());
        assert!(record.value < 1.0);
    }

    // ── The record ───────────────────────────────────────────────────────

    #[test]
    fn every_input_is_recorded_with_its_unit() {
        let record = evaluate("(8.2 mm - 9.0 mm) / 9.0 mm * 100").unwrap();
        assert_eq!(record.inputs, vec!["8.2 mm", "9 mm", "9 mm", "100"]);
    }

    /// A reviewer must be able to follow the working without redoing it.
    #[test]
    fn intermediate_steps_are_recorded_in_order() {
        let record = evaluate("(8.2 mm - 9.0 mm) / 9.0 mm * 100").unwrap();

        assert_eq!(record.steps[0].description, "8.2 mm - 9 mm");
        assert_eq!(record.steps[0].result, "-0.8 mm");
        assert_eq!(record.steps[1].description, "-0.8 mm / 9 mm");
        assert_eq!(record.steps.last().unwrap().result, "-8.8888888889");
    }

    #[test]
    fn the_record_states_its_rounding_and_keeps_the_exact_value() {
        let record = evaluate("1 / 3").unwrap();
        assert_eq!(record.formatted, "0.3333");
        assert!((record.value - 0.333_333_333_333).abs() < 1e-9, "the exact value is kept");
        assert!(record.rounding.contains("significant figures"));
    }

    /// A record's provenance is on the record, not something a reader must know.
    #[test]
    fn every_record_says_it_was_computed_deterministically() {
        assert!(evaluate("1 + 1").unwrap().deterministic);
    }

    // ── Refusing rather than guessing ────────────────────────────────────

    #[test]
    fn division_by_zero_is_refused() {
        assert!(evaluate("5 / 0").unwrap_err().message.contains("Division by zero"));
    }

    #[test]
    fn an_empty_expression_is_refused() {
        assert!(evaluate("   ").unwrap_err().message.contains("nothing to calculate"));
    }

    #[test]
    fn an_unclosed_bracket_is_refused() {
        assert!(evaluate("(1 + 2").unwrap_err().message.contains("never closed"));
    }

    /// A partial reading is worse than a refusal, because it produces a number.
    #[test]
    fn trailing_nonsense_is_refused_rather_than_partly_evaluated() {
        let error = evaluate("2 + 2 more please").unwrap_err();
        assert!(error.message.contains("does not read as a unit"), "{}", error.message);
    }

    /// Prose containing a number is not a calculation, and must not become one.
    #[test]
    fn a_sentence_is_not_mistaken_for_an_expression() {
        for prose in [
            "the wall is 8 mm thick",
            "2 and then some",
            "measured 8.2 mm at four points",
        ] {
            assert!(evaluate(prose).is_err(), "{prose:?} should not evaluate");
        }
    }

    /// But a genuine trailing token still reports as unreadable.
    #[test]
    fn a_stray_symbol_after_a_complete_expression_is_refused() {
        let error = evaluate("2 + 2 )").unwrap_err();
        assert!(error.message.contains("Could not make sense of"), "{}", error.message);
    }

    #[test]
    fn an_error_says_where_it_happened() {
        let error = evaluate("8.2 mm + 9.0 kg").unwrap_err();
        assert!(error.position.is_some());
    }

    #[test]
    fn a_percent_sign_reads_as_a_unit() {
        let record = evaluate("5% + 3%").unwrap();
        assert_eq!(record.unit, "%");
        assert_eq!(record.formatted, "8 %");
    }

    /// Floating-point noise must never reach a document.
    #[test]
    fn the_written_result_carries_no_floating_point_noise() {
        assert_eq!(evaluate("0.1 + 0.2").unwrap().formatted, "0.3");
        assert_eq!(evaluate("3 * 1.1").unwrap().formatted, "3.3");
    }
}
