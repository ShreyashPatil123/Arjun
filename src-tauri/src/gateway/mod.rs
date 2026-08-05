//! Local HTTP gateway — lets external coding tools use Sarathi's loaded model.
//!
//! Sarathi is the engine room: it owns the model, and tools like Claude Code,
//! opencode, and openclaw connect to it rather than loading their own.
//!
//! Two protocol surfaces are served, because the ecosystem is split:
//!
//! - `POST /v1/chat/completions` — OpenAI shape (opencode, openclaw, Cursor)
//! - `POST /v1/messages` — Anthropic shape (Claude Code, via `ANTHROPIC_BASE_URL`)
//!
//! Both funnel into the same [`GenerationScheduler`], so external requests and
//! the desktop app share one model and take turns rather than competing for it.

pub mod anthropic;
pub mod guard;
pub mod openai;
pub mod server;
pub mod state;

pub use server::{start_gateway, GatewayHandle};
pub use state::{GatewayConfig, GatewayState, GatewayStats, ClientActivity};

/// Default port. Deliberately not Ollama's 11434, so both can run at once.
pub const DEFAULT_PORT: u16 = 11435;
