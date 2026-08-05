//! AI Engine Module (Phase 5)
//! Interface for managing AI model inference backends.
//!
//! - `traits`: Data types, enums, and abstract traits
//! - `runtime`: LlamaCpp runtime implementation (GGUF inference)
//! - `manager`: Thread-safe inference state manager (Tauri integration)
//! - `lora_binding`: LoRA adapter caching and live-context binding

pub mod traits;
pub mod runtime;
pub mod manager;
pub mod session;
pub mod lora_binding;
pub mod scheduler;
pub mod vram_planner;

pub use traits::*;
pub use manager::InferenceManager;
pub use session::*;
