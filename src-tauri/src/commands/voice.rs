//! Tauri commands for the voice bridge.
//!
//! See [`crate::voice`] for the module-level documentation; the
//! short version is: this is a push-to-talk bridge, the sidecar runs
//! in `--mode stub` by default, and the real model kicks in when
//! the operator drops the weights into `<app_data_dir>/voice/`.

use std::path::PathBuf;

use serde::Serialize;
use tauri::State;

use crate::voice::{self, SpeakRequest, TranscribeResult};

/// What the sidecar *would* do for a given request, before any
/// audio is processed. The front-end asks the user agent on mount
/// to disable the mic button when neither path is available.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct VoiceStatus {
    pub sidecar_present: bool,
    pub whisper_model_present: bool,
    pub piper_voice_present: bool,
    pub mode: &'static str,
}

/// Reports which voice paths are live in the current installation.
/// Used by the front-end to show a small "STT: stub" / "STT: real" tag
/// next to the microphone button.
#[tauri::command]
pub async fn voice_status(
    app_data_dir: State<'_, PathBuf>,
) -> Result<VoiceStatus, String> {
    Ok(VoiceStatus {
        sidecar_present: voice::sidecar_path().exists(),
        whisper_model_present: voice::has_whisper_model(&app_data_dir),
        piper_voice_present: voice::has_piper_voice(&app_data_dir),
        mode: if voice::has_whisper_model(&app_data_dir) {
            "real-stt"
        } else {
            "stub"
        },
    })
}

/// Transcribes a chunk of audio recorded by the front-end. The
/// audio is the raw bytes from a `MediaRecorder` stream; the sidecar
/// decodes it.
///
/// `stub_only` is exposed for the demo so a presenter can show the
/// bridge plumbing without actually playing audio out loud.
#[tauri::command]
pub async fn voice_transcribe(
    app_data_dir: State<'_, PathBuf>,
    audio: Vec<u8>,
    stub_only: Option<bool>,
) -> Result<TranscribeResult, String> {
    voice::transcribe(&app_data_dir, &audio, stub_only.unwrap_or(false)).await
}

/// Synthesises speech from text. The sidecar writes a WAV file and
/// returns the path; the front-end plays it through an `<audio>`
/// element.
#[tauri::command]
pub async fn voice_speak(
    app_data_dir: State<'_, PathBuf>,
    request: SpeakRequest,
) -> Result<PathBuf, String> {
    voice::speak(&app_data_dir, &request).await
}
