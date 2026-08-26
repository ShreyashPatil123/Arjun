//! Sovereignty — where the "nothing leaves this machine" claim is enforced.
//!
//! - [`mode`]: the two operating modes, and the invariant that keeps confidential
//!   data and network access from ever being enabled together.
//! - [`broker`]: the single outbound chokepoint, and the canary that proves the
//!   controls are live rather than merely configured.
//! - [`observer`]: what the operating system independently says this process is
//!   connected to, so the claim does not rest on ARJUN reporting on itself.

pub mod broker;
pub mod mode;
pub mod observer;

pub use broker::{global_broker, BrokerError, EgressEvent, NetworkBroker};
pub use mode::{OperatingMode, Refusal};
pub use observer::{observe_own_connections, ObservationReport, ObservedConnection};
