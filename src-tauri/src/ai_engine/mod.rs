//! AI Engine Module (Phase 5)
//! Interface for managing AI model inference backends.
//!
//! - `traits`: Data types, enums, and abstract traits
//! - `runtime`: LlamaCpp runtime implementation (GGUF inference)
//! - `manager`: Thread-safe inference state manager (Tauri integration)
//! - `gguf_meta`: Model geometry read from the GGUF header, before loading
//! - `residency`: which model is in VRAM, and when to swap or release it
//! - `activation`: performs the swap, and keeps it from happening mid-task
//! - `orchestrator_state`: explicit model lifecycle (Loading / Warm /
//!   Inference / Unloading / Error) — TODO 3 of the 7-step plan
//! - `orchestrator_path`: resolves the orchestrator GGUF to a real path,
//!   with sha256 verification and a library-scan fallback
//! - `request_context`: per-request bookkeeping for multi-employee
//!   concurrency — TODO 6 of the 7-step plan
//! - `vision_bridge`: takes image paths + a query and produces a structured
//!   description via a local VLM, in the OpenAI-compatible vision schema.
//! - `token_reconciliation`: compares a token estimate against what the
//!   tokenizer actually counted, and keeps the two apart in the record
//! - `ocr_budget`: decides how much of an OCR'd document fits in the prompt

pub mod traits;
pub mod runtime;
pub mod manager;
pub mod session;
pub mod startup;
pub mod scheduler;
pub mod activation;
pub mod residency;
pub mod orchestrator_state;
pub mod orchestrator_path;
pub mod request_context;
pub mod vram_planner;
pub mod gguf_meta;
pub mod vision_bridge;
pub mod ocr_spans;
pub mod ocr_profile;
pub mod ocr_repetition;
pub mod ocr_stream;
pub mod token_reconciliation;
pub mod ocr_budget;

pub use traits::*;
pub use manager::InferenceManager;
pub use session::*;
pub use vision_bridge::{VisionLanguageBridge, VisionRequest, VisionResponse};
pub use orchestrator_state::{
    is_valid_transition, ModelPhase, ModelState, ORCHESTRATOR_COOLDOWN,
};
pub use orchestrator_path::{
    declared_path, resolve_orchestrator_path, OrchestratorPathError, ResolvedOrchestrator,
};
pub use request_context::{RequestContext, RequestRegistry, DEFAULT_SLOT_COUNT};
