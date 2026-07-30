//! Logging and error handling setup

use log::{error, info};
use std::panic;

/// Sets up a panic handler that logs the error before crashing
pub fn setup_panic_handler() {
    panic::set_hook(Box::new(|panic_info| {
        let location = panic_info.location().unwrap();
        let message = match panic_info.payload().downcast_ref::<&str>() {
            Some(s) => *s,
            None => match panic_info.payload().downcast_ref::<String>() {
                Some(s) => &s[..],
                None => "Box<dyn Any>",
            },
        };

        let err_msg = format!("Application crashed! Location: {}:{}, Message: {}", location.file(), location.line(), message);
        error!("{}", err_msg);
        
        // At this point we could potentially also write to a specific crash.log file
        // or trigger an alert, but just logging it through the normal logger ensures
        // it goes to our logdir configured in tauri-plugin-log.
    }));
    
    info!("Panic handler registered");
}
