//! LoRA Module (Phase 5+)
//! Manages LoRA adapter validation, format detection, and future runtime integration.

pub mod convert;
pub mod traits;
pub mod validator;

pub use convert::{convert_adapter, ConversionSummary};
pub use traits::*;
pub use validator::{AdapterRuntimeStatus, AdapterValidationResult};
