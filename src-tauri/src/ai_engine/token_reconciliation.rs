//! Comparing a token estimate against what the tokenizer actually counted.
//!
//! ## Why this exists as its own module
//!
//! Every token figure in ARJUN comes from one of three places, and they are not
//! interchangeable:
//!
//! 1. **The model's own tokenizer** — `llama.cpp`'s `str_to_token`, run against
//!    the real prompt with the real vocabulary. Exact.
//! 2. **A server's reported usage** — `usage.prompt_tokens` off an
//!    OpenAI-compatible response. Exact if the server counted honestly.
//! 3. **A character or word heuristic** — chars ÷ 4, or words × 1.3. A guess.
//!
//! The bug this module was written after is what happens when a (3) is stored
//! in a field whose readers assume (1). Nothing crashes. The number is
//! plausible. It is simply wrong, by an amount nobody can see, and every
//! decision resting on it — whether the next turn fits, which model is
//! cheaper — inherits the error silently.
//!
//! So the rule here is that a reconciliation always names its source, and the
//! variant for "nobody counted" is a variant rather than a zero.
//!
//! ## What "recently performed" means
//!
//! Reconciliation happens on **every model call**, at the point the call
//! returns — not on a timer and not once per conversation. A run that made four
//! tool-loop turns produces four reconciliations. Anything less frequent means
//! the running total is a guess for most of the run's life, which defeats the
//! purpose of keeping one.

use serde::{Deserialize, Serialize};

/// Where a token count came from. Never inferred; always recorded by whoever
/// produced the number.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum TokenSource {
    /// The model's own tokenizer counted it. Exact.
    Tokenizer,
    /// A serving layer reported it in a usage block. Exact if it counted.
    ProviderUsage,
    /// A character or word heuristic. A guess, and labelled as one.
    Estimated,
}

impl TokenSource {
    /// True when the figure is a measurement rather than a guess.
    ///
    /// The screens use this to decide whether to print a `~`. A caller must
    /// never use it to decide whether to *substitute* one source for another.
    pub fn is_measured(self) -> bool {
        matches!(self, TokenSource::Tokenizer | TokenSource::ProviderUsage)
    }

    pub fn label(self) -> &'static str {
        match self {
            TokenSource::Tokenizer => "model tokenizer",
            TokenSource::ProviderUsage => "reported by the server",
            TokenSource::Estimated => "estimated",
        }
    }
}

/// One estimate compared against one measurement.
///
/// Both halves are kept. Storing only the corrected total would make the
/// estimator impossible to improve, because nobody could see how wrong it had
/// been.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Reconciliation {
    /// What the ledger predicted before the call.
    pub estimated: u32,
    /// What was actually counted. `None` when nothing counted it — which is
    /// reported as such rather than filled in from `estimated`.
    pub measured: Option<u32>,
    pub source: TokenSource,
    /// `measured / estimated`. `None` when there is nothing to compare, either
    /// because nothing was measured or because the estimate was zero — a ratio
    /// against zero is not a large drift, it is undefined.
    pub drift_ratio: Option<f64>,
}

impl Reconciliation {
    /// Compares an estimate with a measurement.
    pub fn new(estimated: u32, measured: Option<u32>, source: TokenSource) -> Self {
        let drift_ratio = match measured {
            Some(measured) if estimated > 0 => Some(f64::from(measured) / f64::from(estimated)),
            _ => None,
        };
        Self { estimated, measured, source, drift_ratio }
    }

    /// The figure a running total should be corrected to.
    ///
    /// The measurement when there is one, the estimate otherwise. The caller
    /// still has `source` and `measured` to tell which it got, and the screens
    /// use that to mark the total approximate rather than presenting a
    /// corrected-looking number that was never corrected.
    pub fn authoritative(&self) -> u32 {
        self.measured.unwrap_or(self.estimated)
    }

    /// True when the estimate was off by enough to be worth showing.
    ///
    /// Ten percent. Below that the estimator is doing its job and a warning
    /// would be noise; above it, a person sizing a context window against these
    /// numbers is being misled by a margin that matters.
    pub fn is_significant(&self) -> bool {
        match self.drift_ratio {
            Some(ratio) => (ratio - 1.0).abs() > 0.10,
            None => false,
        }
    }
}

/// The drift between an estimate and a measurement, in words.
///
/// Names the direction because the two are not equally dangerous: an estimate
/// that reads *low* is the one that lets a prompt overflow a window everybody
/// was told had room.
pub fn drift_label(estimated: u32, measured: u32) -> String {
    if estimated == 0 {
        return "no estimate to compare".to_string();
    }
    let ratio = f64::from(measured) / f64::from(estimated);
    let percent = (ratio - 1.0) * 100.0;
    if percent.abs() < 0.5 {
        "within 0.5%".to_string()
    } else if percent > 0.0 {
        format!("estimate read {percent:.0}% low")
    } else {
        format!("estimate read {:.0}% high", percent.abs())
    }
}

/// How much to scale later estimates in the same run, from drift seen so far.
///
/// ## Why this is bounded, and why it needs two samples
///
/// A calibration factor derived from one call is derived from one call's
/// quirks — a turn that was mostly a base64 image tokenizes nothing like the
/// run's prose. Applying that to everything afterwards trades a known small
/// error for an unknown large one.
///
/// The clamp is the same argument at the other end. A factor outside 0.5–2.0
/// does not mean the estimator needs a nudge; it means something is wrong with
/// the comparison — a truncated prompt, or a cached prefix the server counted
/// and we did not. Scaling by it would propagate the fault into every later
/// figure, so it is clamped and the raw drift stays visible in the record.
///
/// `None` means "do not calibrate", which callers must treat as leaving their
/// estimates alone rather than as a factor of zero.
pub fn calibration_factor(samples: &[Reconciliation]) -> Option<f64> {
    let ratios: Vec<f64> = samples.iter().filter_map(|r| r.drift_ratio).collect();
    if ratios.len() < 2 {
        return None;
    }
    let mean = ratios.iter().sum::<f64>() / ratios.len() as f64;
    if !mean.is_finite() || mean <= 0.0 {
        return None;
    }
    Some(mean.clamp(0.5, 2.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Compares ratios with a tolerance.
    ///
    /// These are means of divisions, so exact equality is the wrong assertion —
    /// 1.2 + 1.4 over 2 lands on 1.2999999999999998, and a test that fails on
    /// that is testing IEEE 754 rather than the calibration rule.
    fn close(actual: Option<f64>, expected: f64) -> bool {
        matches!(actual, Some(value) if (value - expected).abs() < 1e-9)
    }

    #[test]
    fn an_uncounted_call_reports_absence_rather_than_zero() {
        let r = Reconciliation::new(500, None, TokenSource::Estimated);
        assert_eq!(r.measured, None, "absence was replaced by a number");
        assert_eq!(r.drift_ratio, None, "a drift was invented with nothing to compare");
        // The running total still has to move forward, and the estimate is all
        // there is — but `source` and `measured` both still say so.
        assert_eq!(r.authoritative(), 500);
        assert!(!r.source.is_measured());
    }

    #[test]
    fn the_measurement_wins_over_the_estimate() {
        let r = Reconciliation::new(400, Some(512), TokenSource::Tokenizer);
        assert_eq!(r.authoritative(), 512);
        assert!(r.source.is_measured());
        assert!(close(r.drift_ratio, 1.28), "{:?}", r.drift_ratio);
    }

    #[test]
    fn a_zero_estimate_yields_no_ratio() {
        // Not an infinite drift: dividing by an estimate of zero is undefined,
        // and a screen reporting "estimate read inf% low" helps nobody.
        let r = Reconciliation::new(0, Some(120), TokenSource::Tokenizer);
        assert_eq!(r.drift_ratio, None);
        assert!(!r.is_significant());
        assert_eq!(r.authoritative(), 120);
    }

    #[test]
    fn small_drift_is_not_flagged_and_large_drift_is() {
        assert!(!Reconciliation::new(1000, Some(1050), TokenSource::Tokenizer).is_significant());
        assert!(Reconciliation::new(1000, Some(1400), TokenSource::Tokenizer).is_significant());
        // Symmetric: an over-estimate by the same margin is equally worth
        // showing, even though it is the safer direction to be wrong in.
        assert!(Reconciliation::new(1000, Some(600), TokenSource::Tokenizer).is_significant());
    }

    #[test]
    fn drift_names_the_direction() {
        assert!(drift_label(1000, 1300).contains("low"), "under-estimate not named");
        assert!(drift_label(1000, 700).contains("high"), "over-estimate not named");
        assert_eq!(drift_label(1000, 1000), "within 0.5%");
        assert_eq!(drift_label(0, 100), "no estimate to compare");
    }

    #[test]
    fn one_sample_does_not_calibrate() {
        let one = [Reconciliation::new(100, Some(130), TokenSource::Tokenizer)];
        assert_eq!(calibration_factor(&one), None, "calibrated from a single call");
    }

    #[test]
    fn two_samples_average() {
        let samples = [
            Reconciliation::new(100, Some(120), TokenSource::Tokenizer),
            Reconciliation::new(100, Some(140), TokenSource::Tokenizer),
        ];
        assert!(close(calibration_factor(&samples), 1.3));
    }

    #[test]
    fn an_absurd_factor_is_clamped_not_applied() {
        // A tenfold drift is not an estimator that needs scaling by ten; it is
        // a comparison that has gone wrong. The clamp keeps the fault from
        // propagating into every later figure.
        let samples = [
            Reconciliation::new(10, Some(1000), TokenSource::Tokenizer),
            Reconciliation::new(10, Some(1000), TokenSource::Tokenizer),
        ];
        assert!(close(calibration_factor(&samples), 2.0));
    }

    #[test]
    fn uncounted_samples_do_not_calibrate() {
        let samples = [
            Reconciliation::new(100, None, TokenSource::Estimated),
            Reconciliation::new(100, None, TokenSource::Estimated),
        ];
        assert_eq!(
            calibration_factor(&samples),
            None,
            "calibrated from calls that measured nothing"
        );
    }
}
