//! What a child sends back, and why it cannot quietly claim success.
//!
//! ## Status is not derived from the payload
//!
//! Requirement 8: a child's failure, timeout or cancellation must be visible to
//! the parent and must never be silently converted into success.
//!
//! The way that goes wrong is subtle. A worker times out having found two of
//! the four passages it was after; the parent sees two passages, folds them in,
//! and the run continues as though the retrieval finished. Nothing lied — the
//! two passages are real — and the answer is nonetheless built on a search that
//! did not complete.
//!
//! So [`ChildStatus`] is a field the manager sets from what actually happened,
//! not something inferred from whether `findings` is empty. A timed-out child
//! **may** carry findings, and [`ChildResult::is_complete`] is still false, and
//! the parent has to decide what to do about a partial result rather than being
//! handed one that looks whole.
//!
//! ## Compact, and referenced
//!
//! Findings carry evidence *references*, not passages. The parent already has a
//! way to resolve a marker; sending the text back would duplicate it into a
//! second place, under a second set of clearance assumptions.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::profile::SchemaKind;

/// How a child ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ChildStatus {
    /// It finished the work it was given.
    Completed,
    /// It ran and could not finish. `detail` on the result says why.
    Failed,
    /// It reached its deadline. May carry partial findings, and is still not a
    /// success.
    TimedOut,
    /// The parent stopped it, or the run it belonged to ended.
    Cancelled,
    /// It was never started: the policy refused it. Distinct from `Failed`
    /// because nothing went wrong — the answer was no.
    Refused,
    /// It ran, did part of the work, and says what it did not do.
    ///
    /// Not a success, and not a failure a retry would fix: the part it did is
    /// real and the part it did not is named in `missing`. Reported separately
    /// so a parent cannot fold three pages of a four-page extraction into a
    /// note as though it had read the fourth.
    Partial,
    /// It could not start or could not finish for want of something this
    /// machine does not have -- a renderer, a container daemon, a model.
    ///
    /// Distinct from `Failed`, which is the work going wrong, and from
    /// `Refused`, which is a policy saying no. A blocked result names the
    /// prerequisite in `missing`, and is the honest state of a deployment that
    /// lacks it; reporting it as either of the others would mislead whoever
    /// decides what to fix.
    Blocked,
}

impl ChildStatus {
    pub const fn as_str(self) -> &'static str {
        match self {
            ChildStatus::Completed => "completed",
            ChildStatus::Failed => "failed",
            ChildStatus::TimedOut => "timed_out",
            ChildStatus::Cancelled => "cancelled",
            ChildStatus::Refused => "refused",
            ChildStatus::Partial => "partial",
            ChildStatus::Blocked => "blocked",
        }
    }

    /// Only one value means the work is done.
    pub const fn is_complete(self) -> bool {
        matches!(self, ChildStatus::Completed)
    }

    /// What a parent should say about a child that ended this way.
    pub const fn describe(self) -> &'static str {
        match self {
            ChildStatus::Completed => "finished",
            ChildStatus::Failed => "did not finish; treat anything it returned as incomplete",
            ChildStatus::TimedOut => {
                "ran out of time; anything it returned is partial and the rest was not looked at"
            }
            ChildStatus::Cancelled => "was stopped before it finished",
            ChildStatus::Refused => "was not started, because it was not permitted",
            ChildStatus::Partial => {
                "did part of the work; what it did not do is listed, and the rest is not done"
            }
            ChildStatus::Blocked => {
                "could not run for want of something this machine does not have, which is named"
            }
        }
    }
}

/// A file a child produced, at the exact version it produced.
///
/// The version and the hash both: a claim about "the approval note" that does
/// not say which revision, and whose bytes, is not checkable afterwards.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ArtifactVersion {
    pub artifact_id: String,
    pub version: u32,
    pub sha256: String,
}

/// The durable event a finding rests on.
///
/// A reference into the run's own event log, which is what a receipt *is*:
/// the record, written as it happened, that a named tool ran and succeeded. Not
/// the tool's output, and not the model's account of it. Plan §3 finding 2 is
/// that workers have been publishing receipts with no event behind them; this
/// is the shape the real one takes, and P02 is where it gets filled in.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ReceiptRef {
    pub run_id: String,
    pub tool: String,
    /// The event's sequence number. Never zero for a real receipt.
    pub event_seq: i64,
}

/// A check run on what a child produced, and how it came out.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ValidationCheck {
    /// Stable id of the check, e.g. `art-docx-02-required-sections`.
    pub check_id: String,
    /// `passed`, `failed` or `blocked`. A string rather than a bool because a
    /// check that could not be run is neither, and a bool would have to lie.
    pub outcome: String,
    pub detail: String,
}

/// A passage or file a finding rests on. A reference, never the text.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EvidenceRef {
    /// The marker the parent knows this passage by, where there is one.
    pub marker: Option<usize>,
    /// The document's content hash.
    pub document_sha256: String,
    pub page: Option<u32>,
    /// A short citation string, for a person reading the trace.
    pub citation: String,
}

/// One thing a child established.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Finding {
    /// What was found, in the child's words. Short: this is a result, not a
    /// report, and the parent writes the report.
    pub statement: String,
    /// What it rests on. A finding with no evidence is permitted and is
    /// reported as such — a worker that found nothing has found nothing, and
    /// inventing a citation for it would be worse.
    pub evidence: Vec<EvidenceRef>,
}

/// What a child hands back.
/// `PartialEq` and deliberately not `Eq`: `confidence` is a float, and two
/// results being "equal" is a comparison for tests rather than a key.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ChildResult {
    pub child_id: String,
    pub profile: String,
    /// Set by the manager from what happened. Never inferred from the payload.
    pub status: ChildStatus,
    /// The shape this was asked for. A result whose schema does not match the
    /// packet's is refused by the parent rather than folded in.
    pub schema: SchemaKind,
    pub findings: Vec<Finding>,
    /// How sure the child is, 0.0 to 1.0. Not a model probability — a worker's
    /// own account of how much it had to infer.
    pub confidence: f32,
    /// What it could not establish. Named individually, because "some
    /// uncertainty" is not actionable and "page 4 could not be read" is.
    pub uncertainty: Vec<String>,
    /// Present when the status is not `Completed`, in words a person reads.
    pub detail: Option<String>,
    /// How many turns it actually took.
    pub turns_used: u32,
    /// Items this child committed to the task's shared memory, by id.
    ///
    /// The part a sibling can act on. A finding in this payload is something
    /// the parent read; an id here is something *anybody* on the task can go
    /// and read for themselves, at a revision — which is what makes one
    /// worker's output reach another without a transcript passing between them.
    ///
    /// Defaulted so a result recorded before this existed still parses.
    #[serde(default)]
    pub published: Vec<String>,
    /// Files this child produced, at exact versions.
    ///
    /// Defaulted, like every field below, so a result recorded before these
    /// existed still parses -- the idempotency ledger replays settled results
    /// for the life of a task.
    #[serde(default)]
    pub artifacts: Vec<ArtifactVersion>,
    /// The events the findings rest on.
    #[serde(default)]
    pub receipts: Vec<ReceiptRef>,
    /// Checks run on what was produced.
    #[serde(default)]
    pub validation: Vec<ValidationCheck>,
    /// What was not done, for a `Partial` result, or the prerequisite that was
    /// missing, for a `Blocked` one. Empty for a completed result.
    #[serde(default)]
    pub missing: Vec<String>,
    /// SHA-256 over the findings and status, so the parent's record of what
    /// came back can be checked against the child's.
    pub result_hash: String,
    pub finished_at: DateTime<Utc>,
}

impl ChildResult {
    /// A result for a child that produced findings and finished.
    pub fn completed(
        child_id: impl Into<String>,
        profile: impl Into<String>,
        schema: SchemaKind,
        findings: Vec<Finding>,
        confidence: f32,
        uncertainty: Vec<String>,
        turns_used: u32,
    ) -> Self {
        Self::sealed(
            child_id.into(),
            profile.into(),
            ChildStatus::Completed,
            schema,
            findings,
            confidence.clamp(0.0, 1.0),
            uncertainty,
            None,
            turns_used,
        )
    }

    /// A result for a child that did not finish.
    ///
    /// Takes whatever findings it had. Keeping them is right — two passages
    /// found before a timeout are two real passages — and the status is what
    /// stops them being read as a completed search.
    pub fn ended(
        child_id: impl Into<String>,
        profile: impl Into<String>,
        status: ChildStatus,
        schema: SchemaKind,
        findings: Vec<Finding>,
        detail: impl Into<String>,
        turns_used: u32,
    ) -> Self {
        debug_assert!(
            !status.is_complete(),
            "`ended` is for a child that did not finish; use `completed`"
        );
        Self::sealed(
            child_id.into(),
            profile.into(),
            status,
            schema,
            findings,
            // A child that did not finish is not confident. Set here rather
            // than taken from the worker, so a worker cannot report high
            // confidence in a result it did not finish producing.
            0.0,
            Vec::new(),
            Some(detail.into()),
            turns_used,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn sealed(
        child_id: String,
        profile: String,
        status: ChildStatus,
        schema: SchemaKind,
        findings: Vec<Finding>,
        confidence: f32,
        uncertainty: Vec<String>,
        detail: Option<String>,
        turns_used: u32,
    ) -> Self {
        let result_hash = hash_of(&status, schema, &findings);
        Self {
            child_id,
            profile,
            status,
            schema,
            findings,
            // Filled in by the worker after it publishes: `sealed` is about the
            // shape a result must have, and where its findings landed is known
            // only once they have.
            published: Vec::new(),
            artifacts: Vec::new(),
            receipts: Vec::new(),
            validation: Vec::new(),
            missing: Vec::new(),
            confidence,
            uncertainty,
            detail,
            turns_used,
            result_hash,
            finished_at: Utc::now(),
        }
    }

    /// Records a file this child produced, and re-seals.
    pub fn with_artifact(mut self, artifact: ArtifactVersion) -> Self {
        self.artifacts.push(artifact);
        self.reseal()
    }

    /// Records the event a finding rests on, and re-seals.
    ///
    /// Refuses a receipt that names no event. Sequence numbers in the event
    /// log start at 1, so `event_seq <= 0` is a receipt with nothing behind it
    /// -- the exact shape plan §3 finding 2 found workers publishing -- and a
    /// result carrying one would present an unbacked claim as a verified one.
    pub fn with_receipt(mut self, receipt: ReceiptRef) -> Result<Self, String> {
        if receipt.event_seq <= 0 || receipt.run_id.trim().is_empty() || receipt.tool.trim().is_empty()
        {
            return Err(format!(
                "a receipt must name the run, the tool and a recorded event; got run {:?}, tool \
                 {:?}, event {}",
                receipt.run_id, receipt.tool, receipt.event_seq
            ));
        }
        self.receipts.push(receipt);
        Ok(self.reseal())
    }

    /// Records a check on what was produced, and re-seals.
    pub fn with_validation(mut self, check: ValidationCheck) -> Self {
        self.validation.push(check);
        self.reseal()
    }

    /// Names what a partial or blocked result did not do, and re-seals.
    pub fn with_missing(mut self, what: impl Into<String>) -> Self {
        self.missing.push(what.into());
        self.reseal()
    }

    /// Recomputes the hash over everything the result now carries.
    ///
    /// The new fields enter the hash only when they are non-empty, so a result
    /// that carries none of them hashes exactly as it did before they existed
    /// -- which is what keeps every hash already recorded in an event log
    /// verifiable.
    fn reseal(mut self) -> Self {
        self.result_hash = hash_with_contract(
            &self.status,
            self.schema,
            &self.findings,
            &self.artifacts,
            &self.receipts,
            &self.validation,
            &self.missing,
        );
        self
    }

    /// Whether the parent may treat this as the work being done.
    pub fn is_complete(&self) -> bool {
        self.status.is_complete()
    }

    /// Whether this answers the packet it came back from.
    ///
    /// Checked by the parent rather than trusted: a result whose schema does
    /// not match what was asked for is a worker answering a different question,
    /// and folding it in would put an extraction where a calculation belongs.
    pub fn answers(&self, packet: &super::packet::ChildTaskPacket) -> bool {
        self.child_id == packet.child_id && self.schema == packet.required_schema
    }

    /// One line for the parent's trace.
    pub fn describe(&self) -> String {
        let head = format!("{} {}", self.profile, self.status.describe());
        match (&self.detail, self.findings.len()) {
            (Some(detail), 0) => format!("{head}: {detail}"),
            (Some(detail), n) => format!("{head}: {detail} ({n} partial finding(s) kept)"),
            (None, n) => format!("{head}, with {n} finding(s)"),
        }
    }
}

/// The seal over what a child established.
///
/// Over the status as well as the findings, so a record that kept the findings
/// and changed the status does not match — which is exactly the alteration
/// requirement 8 is about.
/// The hash, over the whole contract.
///
/// Identical to [`hash_of`] when the four newer lists are empty. Each list is
/// fed in only when it has something in it, behind its own tag, so adding a
/// field to a result that never uses it cannot move a hash that was already
/// recorded.
fn hash_with_contract(
    status: &ChildStatus,
    schema: SchemaKind,
    findings: &[Finding],
    artifacts: &[ArtifactVersion],
    receipts: &[ReceiptRef],
    validation: &[ValidationCheck],
    missing: &[String],
) -> String {
    if artifacts.is_empty() && receipts.is_empty() && validation.is_empty() && missing.is_empty() {
        return hash_of(status, schema, findings);
    }
    let mut hasher = Sha256::new();
    hasher.update(hash_of(status, schema, findings).as_bytes());
    if !artifacts.is_empty() {
        hasher.update(b"\x1cartifacts");
        for artifact in artifacts {
            hasher.update(artifact.artifact_id.as_bytes());
            hasher.update(b"\x1d");
            hasher.update(artifact.version.to_string().as_bytes());
            hasher.update(b"\x1d");
            hasher.update(artifact.sha256.as_bytes());
            hasher.update(b"\x1f");
        }
    }
    if !receipts.is_empty() {
        hasher.update(b"\x1creceipts");
        for receipt in receipts {
            hasher.update(receipt.run_id.as_bytes());
            hasher.update(b"\x1d");
            hasher.update(receipt.tool.as_bytes());
            hasher.update(b"\x1d");
            hasher.update(receipt.event_seq.to_string().as_bytes());
            hasher.update(b"\x1f");
        }
    }
    if !validation.is_empty() {
        hasher.update(b"\x1cvalidation");
        for check in validation {
            hasher.update(check.check_id.as_bytes());
            hasher.update(b"\x1d");
            hasher.update(check.outcome.as_bytes());
            hasher.update(b"\x1f");
        }
    }
    if !missing.is_empty() {
        hasher.update(b"\x1cmissing");
        for what in missing {
            hasher.update(what.as_bytes());
            hasher.update(b"\x1f");
        }
    }
    format!("{:x}", hasher.finalize())
}

fn hash_of(status: &ChildStatus, schema: SchemaKind, findings: &[Finding]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(status.as_str().as_bytes());
    hasher.update(b"\x1f");
    hasher.update(schema.as_str().as_bytes());
    hasher.update(b"\x1f");
    for finding in findings {
        hasher.update(finding.statement.as_bytes());
        hasher.update(b"\x1e");
        for evidence in &finding.evidence {
            hasher.update(evidence.document_sha256.as_bytes());
            hasher.update(b"\x1d");
        }
        hasher.update(b"\x1f");
    }
    format!("{:x}", hasher.finalize())
}
