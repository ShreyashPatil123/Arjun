//! AI Engine Module (Phase 5)
//! Interface for managing AI model inference backends.
//!
//! - `traits`: Data types, enums, and abstract traits
//! - `runtime`: LlamaCpp runtime implementation (GGUF inference)
//! - `manager`: Thread-safe inference state manager (Tauri integration)

pub mod traits;
pub mod runtime;
pub mod manager;
pub mod session;

pub use traits::*;
pub use manager::InferenceManager;
pub use session::*;
