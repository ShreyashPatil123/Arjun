//! AI Engine Module (Phase 5)
//! Interface for managing AI model inference backends.
//!
//! - `traits`: Data types, enums, and abstract traits
//! - `runtime`: LlamaCpp runtime implementation (GGUF inference)
//! - `manager`: Thread-safe inference state manager (Tauri integration)
//! - `lora_binding`: LoRA adapter caching and live-context binding
//! - `gguf_meta`: Model geometry read from the GGUF header, before loading
//! - `residency`: which model is in VRAM, and when to swap or release it
//! - `activation`: performs the swap, and keeps it from happening mid-task

pub mod traits;
pub mod runtime;
pub mod manager;
pub mod session;
pub mod lora_binding;
pub mod scheduler;
pub mod activation;
pub mod residency;
pub mod vram_planner;
pub mod gguf_meta;

pub use traits::*;
pub use manager::InferenceManager;
pub use session::*;
