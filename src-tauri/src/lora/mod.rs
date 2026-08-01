//! LoRA Module (Phase 5+)
//! Manages LoRA adapter validation, format detection, and future runtime integration.

pub mod traits;
pub mod validator;

pub use traits::*;
pub use validator::{AdapterRuntimeStatus, AdapterValidationResult};
