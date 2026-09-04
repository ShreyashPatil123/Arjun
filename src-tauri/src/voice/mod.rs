//! Voice interface — push-to-talk STT/TTS bridge.
//!
//! ## Honest scope
//!
//! The voice interface as shipped in this build is a *push-to-talk bridge*:
//!
//! - The React UI shows a microphone button. The user holds the button
//!   down, the front-end records audio with `MediaRecorder`, and on
//!   release the front-end posts the audio to a Tauri command
//!   `voice_transcribe`.
//! - The Rust command invokes the `voice` Python sidecar (see
//!   `sidecars/voice_sidecar/`) and returns the transcript.
//! - The reverse direction — text-to-speech — is a separate command
//!   `voice_speak` that the agent calls when it wants to read the
//!   reply aloud.
//!
//! ## What is *not* shipped
//!
//! - **No wake-word detection** ("Hey Arjun"). The continuous-audio
//!   path requires an always-on local model, which is a multi-day
//!   integration. The push-to-talk button is the honest version of
//!   "voice" in a 5-minute SIH demo.
//! - **No bundled Whisper weights**. The sidecar is launched in
//!   `--stub` mode by default and returns a placeholder transcript.
//!   To enable real transcription, the operator places a `ggml-tiny.en.bin`
//!   Whisper model in `<app_data_dir>/voice/`. The presence of the
//!   file is the flag that switches from stub to real.
//! - **No bundled Piper voice**. Same pattern: a `en_US-lessac-medium.onnx`
//!   file in the same directory switches the synthesizer from
//!   silent to active.
//!
//! The reason for the file-presence pattern is that the SIH venue
//! may not have a working microphone, and a `voice_transcribe` that
//! crashes because the audio device is missing is a worse demo than
//! one that returns a placeholder. The stub is the documented
//! fallback.

use std::path::{Path, PathBuf};
use std::process::Stdio;

use serde::{Deserialize, Serialize};
use tokio::io::AsyncWriteExt;
use tokio::process::Command;

#[cfg(target_os = "windows")]
#[cfg(target_os = "windows")]
const CREATE_NO_WINDOW: u32 = 0x08000000;

/// The result of a `voice_transcribe` call.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TranscribeResult {
    /// The transcribed text, in the language the model is configured
    /// for. May be empty if the audio was silence.
    pub text: String,
    /// True when the sidecar ran in stub mode (no model on disk).
    /// A reviewer reading the result knows what to expect.
    pub stub: bool,
    /// True when the sidecar had a model loaded and ran real STT.
    pub real: bool,
    /// The model's confidence in the transcription, when the sidecar
    /// provides one. `None` for the stub.
    pub confidence: Option<f32>,
    /// The number of milliseconds of audio the sidecar processed.
    pub audio_ms: u64,
    /// The model that ran. `"stub"` for the placeholder, `"whisper-tiny"`
    /// for the real Whisper tiny model.
    pub model_id: String,
}

/// The shape of a `voice_speak` call. The synthesizer is told what
/// to say, in what language, and at what speed. The command returns
/// the path of the audio file the sidecar wrote.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SpeakRequest {
    pub text: String,
    pub language: String,
    pub speed: f32,
}

/// Resolves the directory the voice sidecar reads models from. The
/// sidecar and the Tauri command agree on this path so the operator
/// only has to drop the model files in one place.
pub fn voice_dir(app_data_dir: &Path) -> PathBuf {
    app_data_dir.join("voice")
}

/// Returns true if a Whisper model is on disk. The check is the
/// *existence* of the file, not its validity — the sidecar
/// validates the model when it loads it.
pub fn has_whisper_model(app_data_dir: &Path) -> bool {
    voice_dir(app_data_dir).join("ggml-tiny.en.bin").exists()
}

/// Returns true if a Piper voice is on disk. Same pattern.
pub fn has_piper_voice(app_data_dir: &Path) -> bool {
    voice_dir(app_data_dir).join("en_US-lessac-medium.onnx").exists()
}

/// Builds the path to the Python sidecar script. The script lives
/// next to the other sidecars; the path is resolved relative to the
/// crate root so the build keeps working when the binary is moved.
pub fn sidecar_path() -> PathBuf {
    // The sidecar is a Python script that the Tauri runtime launches
    // directly. Resolved through `deployment`, which tries the installer's
    // resource directory before the checkout; the two relative paths this
    // replaced were interpreted against the working directory, which under an
    // installed build is not the repository and often not even predictable.
    //
    // Still returns a `PathBuf` rather than a `Result` because
    // `commands::voice` asks this for `.exists()` to report whether voice is
    // available — "not installed" is a status this feature reports rather than
    // an error it raises.
    crate::deployment::require_path("voice-sidecar").unwrap_or_else(|_| {
        crate::deployment::dependency("voice-sidecar")
            .bundle_path
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("voice_bridge.py"))
    })
}

/// Transcribe a chunk of audio. The audio is the raw bytes from a
/// webm/opus MediaRecorder stream; the sidecar handles decoding.
///
/// `stub_only` forces the stub path even when a model is on disk;
/// it is exposed for the demo so a presenter can show the bridge
/// plumbing without actually playing the audio out loud.
pub async fn transcribe(
    app_data_dir: &Path,
    audio: &[u8],
    stub_only: bool,
) -> Result<TranscribeResult, String> {
    let sidecar = sidecar_path();
    if !sidecar.exists() {
        return Ok(TranscribeResult {
            text: String::new(),
            stub: true,
            real: false,
            confidence: None,
            audio_ms: 0,
            model_id: "stub-no-sidecar".to_string(),
        });
    }
    let has_model = !stub_only && has_whisper_model(app_data_dir);
    let mut cmd = Command::new(crate::deployment::program("python"));
    cmd.arg(&sidecar)
        .arg("transcribe")
        .arg("--stdin")
        .arg("--model-dir")
        .arg(voice_dir(app_data_dir))
        .arg("--mode")
        .arg(if has_model { "real" } else { "stub" })
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    // The Python sidecar is a console application; without
    // CREATE_NO_WINDOW, Windows pops a terminal window every time
    // the user activates push-to-talk. Off-Windows this block is
    // a no-op and the comment is the only difference.
    #[cfg(target_os = "windows")]
    {
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    let mut child = cmd.spawn().map_err(|e| format!("could not launch voice sidecar: {e}"))?;
    if let Some(mut stdin) = child.stdin.take() {
        stdin
            .write_all(audio)
            .await
            .map_err(|e| format!("could not write audio to sidecar: {e}"))?;
    }
    let out = child
        .wait_with_output()
        .await
        .map_err(|e| format!("sidecar did not complete: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "sidecar exited with code {:?}: {}",
            out.status.code(),
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    let parsed: TranscribeResult = serde_json::from_slice(&out.stdout)
        .map_err(|e| format!("sidecar returned malformed JSON: {e}"))?;
    Ok(parsed)
}

/// Synthesise speech from text. Returns the path of the audio file
/// the sidecar wrote. The caller (the front-end) plays the file
/// through an `<audio>` element.
pub async fn speak(
    app_data_dir: &Path,
    request: &SpeakRequest,
) -> Result<PathBuf, String> {
    let sidecar = sidecar_path();
    if !sidecar.exists() {
        return Err("voice sidecar is not installed in this build".to_string());
    }
    let has_model = has_piper_voice(app_data_dir);
    let out_path = voice_dir(app_data_dir).join("last_reply.wav");
    let mut cmd = Command::new(crate::deployment::program("python"));
    cmd.arg(&sidecar)
        .arg("speak")
        .arg("--model-dir")
        .arg(voice_dir(app_data_dir))
        .arg("--out")
        .arg(&out_path)
        .arg("--mode")
        .arg(if has_model { "real" } else { "stub" })
        .arg("--text")
        .arg(&request.text)
        .arg("--language")
        .arg(&request.language)
        .arg("--speed")
        .arg(request.speed.to_string())
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    #[cfg(target_os = "windows")]
    {
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    let out = cmd
        .output()
        .await
        .map_err(|e| format!("could not launch voice sidecar: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "sidecar exited with code {:?}: {}",
            out.status.code(),
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    Ok(out_path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn voice_dir_is_inside_app_data() {
        let p = voice_dir(Path::new("/tmp/app"));
        assert_eq!(p, PathBuf::from("/tmp/app/voice"));
    }

    #[test]
    fn sidecar_path_resolves_without_panicking() {
        // We do not assert on the path; we only assert that the
        // function does not panic when neither candidate exists.
        // The candidate list is a development convenience; a build
        // that strips the sidecar still gets a deterministic path.
        let _ = sidecar_path();
    }

    #[test]
    fn transcribe_result_serialization_round_trip() {
        let r = TranscribeResult {
            text: "hello world".to_string(),
            stub: true,
            real: false,
            confidence: None,
            audio_ms: 1500,
            model_id: "stub".to_string(),
        };
        let j = serde_json::to_string(&r).unwrap();
        let back: TranscribeResult = serde_json::from_str(&j).unwrap();
        assert_eq!(r.text, back.text);
        assert_eq!(r.model_id, back.model_id);
    }
}
