//! Commands behind the document scan view.
//!
//! The slider in the UI has to describe what it will actually do — which
//! weight file a stop loads, and how much of the page the model will be
//! allowed to see. Those numbers live in [`crate::ai_engine::ocr_profile`],
//! and this command hands them to the frontend rather than letting the UI
//! keep a second copy. A slider whose labels disagree with the profiles that
//! run is worse than no slider: it reports a configuration nobody is using.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter, Manager, State};

use crate::ai_engine::ocr_profile::{
    to_page, CoordSpace, OcrDetent, OcrTier, PageBox, PageGeometry,
};
use crate::ai_engine::ocr_spans::{OcrEvent, RawBox};
use crate::ai_engine::ocr_stream::stream_ocr;
use crate::agent_runtime::stages::StageTag;
use crate::registry::ModelRegistry;
use crate::serving::ModelServers;

/// One slider stop, as the UI needs to render it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OcrDetentInfo {
    pub detent: OcrDetent,
    pub label: String,
    pub tier: OcrTier,
    pub tier_label: String,
    pub max_image_tokens: u32,
    pub max_decode_tokens: u32,
}

/// The stops, fastest first.
pub fn detent_info() -> Vec<OcrDetentInfo> {
    OcrDetent::ALL
        .iter()
        .map(|detent| {
            let profile = detent.profile();
            OcrDetentInfo {
                detent: *detent,
                label: detent.label().to_string(),
                tier: profile.tier,
                tier_label: profile.tier.label().to_string(),
                max_image_tokens: profile.max_image_tokens,
                max_decode_tokens: profile.max_decode_tokens,
            }
        })
        .collect()
}

#[tauri::command]
pub fn get_ocr_detents() -> Vec<OcrDetentInfo> {
    detent_info()
}

/// The coordinate convention this build's model reports in.
///
/// **Measured, not assumed.** The same page was read at 1000x1400 and at
/// 731x1024; the emitted boxes were identical to within a pixel
/// (`title [77, 51, 723, 86]` vs `[77, 52, 723, 86]`). Boxes that do not move
/// with the input size are normalised, not input pixels — that comparison is
/// the whole discriminator, and it is encoded as a test in
/// [`crate::ai_engine::ocr_profile`].
///
/// Cross-checked against known ink positions: the page's bottom-right marker
/// sits at y=1299 of 1400, and the model reported y=923, where normalisation
/// predicts 1299/1400*999 = 927.
const CALIBRATED_COORD_SPACE: Option<CoordSpace> = Some(CoordSpace::Normalised);

/// Width and height from a PNG's IHDR chunk.
///
/// Read directly rather than pulling in an image decoder: the overlay only
/// needs the page's dimensions, and the header carries them at a fixed offset
/// (8-byte signature, 4-byte length, `IHDR`, then two big-endian u32s).
fn png_dimensions(path: &std::path::Path) -> Result<(u32, u32), String> {
    let bytes =
        std::fs::read(path).map_err(|e| format!("could not read {}: {e}", path.display()))?;
    if bytes.len() < 24 || &bytes[..8] != b"\x89PNG\r\n\x1a\n" || &bytes[12..16] != b"IHDR" {
        return Err(format!("{} is not a PNG", path.display()));
    }
    let w = u32::from_be_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]);
    let h = u32::from_be_bytes([bytes[20], bytes[21], bytes[22], bytes[23]]);
    if w == 0 || h == 0 {
        return Err(format!("{} reports a zero dimension", path.display()));
    }
    Ok((w, h))
}

/// A rendered page, ready for the scan view to display.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PageImage {
    /// A `data:` URI.
    ///
    /// The page lives in app data, which the webview cannot reach by path,
    /// and enabling Tauri's asset protocol would open a filesystem route for
    /// the sake of one image. The CSP already allows `data:` for images, and
    /// the vision bridge sends pages to the model the same way — so the page
    /// is handed over as bytes rather than as a path.
    pub data_url: String,
    /// The overlay's coordinate space. Read from the file rather than assumed,
    /// so a page that is not 1000x1400 still gets boxes in the right place.
    pub width: u32,
    pub height: u32,
}

/// Loads one rendered page for display.
#[tauri::command]
pub fn get_page_image(
    app: AppHandle,
    document_sha256: String,
    page: u32,
) -> Result<PageImage, String> {
    let path = page_image_path(&app, &document_sha256, page)?;
    let (width, height) = png_dimensions(&path)?;
    let bytes =
        std::fs::read(&path).map_err(|e| format!("could not read {}: {e}", path.display()))?;
    Ok(PageImage {
        data_url: format!(
            "data:image/png;base64,{}",
            base64::Engine::encode(&base64::engine::general_purpose::STANDARD, &bytes)
        ),
        width,
        height,
    })
}

/// The largest attachment that will be read.
///
/// A page scan is a few hundred kilobytes; anything approaching this is not a
/// document. Bounded here rather than at the model, because a refusal the user
/// can read beats a request that dies inside llama-server.
const MAX_ATTACHMENT_BYTES: usize = 24 * 1024 * 1024;

/// One file the user attached to a chat turn.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ChatAttachment {
    /// The name it arrived under, for display and for the prompt. Never used
    /// as a path — the bytes are addressed by their own hash.
    pub name: String,
    pub mime: String,
    /// Base64 of the file itself. The bytes cross the boundary, not a path:
    /// the webview has no filesystem the backend could re-open.
    pub data_base64: String,
}

/// What reading one attachment produced.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AttachmentRead {
    pub name: String,
    /// Content address of the stored bytes. Two users attaching the same file
    /// converge on one copy, and the id cannot be confused between runs.
    pub sha256: String,
    /// What the OCR model actually read. Never synthesised.
    pub text: String,
    /// Which local path handled it: `image`, `pdf-scan`, `pdf-text`, `docx`…
    pub kind: String,
    /// How many pages were read. One for an image.
    pub pages: u32,
    /// The OCR model that read it, or `None` when no model was needed
    /// because the file already carried its text.
    ///
    /// Reported rather than inferred from `kind`: the routing reasons shown
    /// to the person name this model, and a label that disagrees with what
    /// ran is worse than no label.
    pub ocr_model_id: Option<String>,
    /// The slider stop the read actually ran at.
    pub ocr_detent: Option<OcrDetent>,
}

fn attachment_extension(mime: &str) -> Option<&'static str> {
    match mime {
        "image/png" => Some("png"),
        "image/jpeg" | "image/jpg" => Some("jpg"),
        "image/webp" => Some("webp"),
        _ => None,
    }
}

/// What kind of handling a file needs.
///
/// Routing by type is the whole design. Forcing a PDF through OCR would
/// render pages to read text that is already in the file — slower, lossier,
/// and pointless GPU work. Only images and scans reach the vision model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AttachmentKind {
    /// Straight to the OCR model, unchanged from the path that already works.
    Image(&'static str),
    /// Through the local extractor first; it decides whether OCR is needed.
    Document(&'static str),
}

/// What will happen to a file, decided before a byte of it is sent anywhere.
///
/// The composer shows this so "an OCR model will read this" is visible while
/// the person is still typing, rather than being discovered afterwards from a
/// progress line. It is the same [`validate_attachment`] the run itself uses,
/// so a plan that says a file is unreadable and a run that accepts it cannot
/// disagree.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AttachmentPlan {
    pub name: String,
    /// `image` | `document` | `rejected`.
    pub route: &'static str,
    /// True when a vision model has to look at the page. A PDF carrying its
    /// own text layer is `false` — rendering it to pixels to read back text
    /// the file was already holding is pointless GPU work.
    ///
    /// For a document this is *possible* OCR, not certain: whether a PDF is a
    /// scan is only known once the extractor has opened it.
    pub needs_ocr: bool,
    /// Why, in the person's terms. Shown verbatim.
    pub explanation: String,
    /// Set when the file cannot be read at all; the same sentence the run
    /// would have refused with.
    pub refusal: Option<String>,
}

/// The plan for one file, from its name and MIME type alone.
pub fn plan_attachment(name: &str, mime: &str) -> AttachmentPlan {
    // A length of one: the size gate is a separate question and the composer
    // has not weighed the bytes. Only the type decision is being made here.
    match validate_attachment(name, mime, 1) {
        Ok(AttachmentKind::Image(_)) => AttachmentPlan {
            name: name.to_string(),
            route: "image",
            needs_ocr: true,
            explanation: format!(
                "{name} is an image, so the document-OCR model reads it on this device before the answer is composed."
            ),
            refusal: None,
        },
        Ok(AttachmentKind::Document(ext)) => {
            let scan_possible = ext == "pdf";
            AttachmentPlan {
                name: name.to_string(),
                route: "document",
                needs_ocr: scan_possible,
                explanation: if scan_possible {
                    format!(
                        "{name} is a PDF. Its text layer is read directly; if there is none it is a scan, and each page goes to the document-OCR model."
                    )
                } else {
                    format!(
                        "{name} carries its own text, so it is extracted locally and no model is needed to read it."
                    )
                },
                refusal: None,
            }
        }
        Err(refusal) => AttachmentPlan {
            name: name.to_string(),
            route: "rejected",
            needs_ocr: false,
            explanation: refusal.clone(),
            refusal: Some(refusal),
        },
    }
}

/// One file the composer is about to send, named and typed.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AttachmentDescriptor {
    pub name: String,
    pub mime: String,
}

/// What the attached files will be routed to, before the turn is sent.
#[tauri::command]
pub fn preview_attachment_routing(files: Vec<AttachmentDescriptor>) -> Vec<AttachmentPlan> {
    files
        .iter()
        .map(|f| plan_attachment(&f.name, &f.mime))
        .collect()
}

/// Extensions the local extractor can turn into text without any model.
const DOCUMENT_SUFFIXES: &[&str] = &[
    "pdf", "txt", "md", "markdown", "csv", "json", "log", "tsv", "docx", "xlsx",
];

/// Everything about an attachment that can be judged before touching disk.
///
/// Split out so the limits are testable without a running app: the refusal
/// messages are what the person actually sees, and a silent acceptance here
/// is how an unreadable file becomes an answer about a document nobody read.
fn validate_attachment(name: &str, mime: &str, len: usize) -> Result<AttachmentKind, String> {
    let suffix = name
        .rsplit('.')
        .next()
        .filter(|s| *s != name)
        .unwrap_or("")
        .to_ascii_lowercase();

    // MIME first, because the browser knows better than the name; the suffix
    // is the fallback for picks where the browser reported nothing.
    let kind = if let Some(ext) = attachment_extension(mime) {
        AttachmentKind::Image(ext)
    } else if let Some(ext) = DOCUMENT_SUFFIXES.iter().find(|e| **e == suffix) {
        AttachmentKind::Document(ext)
    } else {
        return Err(format!(
            "{name} is a {mime} — ARJUN reads PDF, Word, Excel, text, Markdown, CSV, JSON and image files."
        ));
    };

    if len == 0 {
        return Err(format!("{name} is empty."));
    }
    if len > MAX_ATTACHMENT_BYTES {
        return Err(format!(
            "{name} is {:.1} MB; the limit is {} MB.",
            len as f64 / 1_048_576.0,
            MAX_ATTACHMENT_BYTES / 1_048_576
        ));
    }
    Ok(kind)
}

/// Where the one-shot document extractor lives.
fn extractor_script() -> Option<PathBuf> {
    let cwd = std::env::current_dir().unwrap_or_default();
    let rel = ["sidecars", "document_sidecar", "attachment_extract.py"];
    let mut candidates = vec![cwd.iter().collect::<PathBuf>()];
    candidates.clear();
    candidates.push(rel.iter().collect());
    candidates.push(cwd.join(rel.iter().collect::<PathBuf>()));
    candidates.push(cwd.join("src-tauri").join(rel.iter().collect::<PathBuf>()));
    if let Some(parent) = cwd.parent() {
        candidates.push(parent.join(rel.iter().collect::<PathBuf>()));
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            candidates.push(dir.join(rel.iter().collect::<PathBuf>()));
        }
    }
    candidates.into_iter().find(|p| p.exists())
}

/// What the extractor reported about one file.
#[derive(Debug, Clone, Deserialize)]
struct Extracted {
    kind: String,
    #[serde(default)]
    text: String,
    #[serde(default)]
    pages: u32,
    #[serde(default, rename = "pageImages")]
    page_images: Vec<PathBuf>,
    #[serde(default)]
    truncated: bool,
    #[serde(default)]
    error: Option<String>,
}

/// What the UI shows while a document is being read.
///
/// Every field is a fact the backend actually has. `page`/`pages` are filled
/// in only once the extractor has counted them, so "Reading page 2 of 6" is
/// never a guess — before that, the phase alone is shown.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AttachmentProgress {
    pub name: String,
    /// `reading` | `preparing` | `extracting` | `understanding` | `done`
    pub phase: &'static str,
    pub page: Option<u32>,
    pub pages: Option<u32>,
    /// Which local path handled it, so the chip can say how it was read.
    pub kind: Option<String>,
    /// The turn this read belongs to.
    ///
    /// This channel used to carry only a filename, which was enough while one
    /// window read one file at a time and wrong the moment it did not: a
    /// progress line has to land on the turn that asked for the read, and a
    /// name cannot say which turn that is. Carried as the caller's own ids
    /// because the read happens before the run has one of its own.
    pub correlation_id: Option<String>,
    pub message_id: Option<String>,
    pub conversation_id: Option<String>,
}

fn progress(
    app: &AppHandle,
    tag: &StageTag,
    name: &str,
    phase: &'static str,
    page: Option<u32>,
    pages: Option<u32>,
    kind: Option<String>,
) {
    let _ = app.emit(
        "attachment:progress",
        AttachmentProgress {
            name: name.to_string(),
            phase,
            page,
            pages,
            kind,
            correlation_id: tag.correlation_id.clone(),
            message_id: tag.message_id.clone(),
            conversation_id: tag.conversation_id.clone(),
        },
    );
}

/// What the OCR model is doing to an attachment right now, character by
/// character.
///
/// A phase label ("Understanding document…") says a model is busy; it does
/// not show the reading. These events carry the model's own output as it
/// arrives, so the person watches the page being transcribed and can tell a
/// good read from a bad one while it happens rather than afterwards.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase", tag = "event")]
pub enum AttachmentOcrEvent {
    /// The model committed to a region and said what kind of thing it is.
    #[serde(rename_all = "camelCase")]
    Region {
        name: String,
        page: u32,
        index: usize,
        /// `title`, `text`, `table`, `figure`, `footer` — the model's label.
        label: String,
    },
    /// Transcribed characters. `index` is the region they belong to, or
    /// `None` for a line the model did not open a region for.
    #[serde(rename_all = "camelCase")]
    Text {
        name: String,
        page: u32,
        index: Option<usize>,
        delta: String,
    },
    /// One page finished, with what it cost. Emitted per page so a six-page
    /// scan shows six completions rather than one at the very end.
    #[serde(rename_all = "camelCase")]
    Page {
        name: String,
        page: u32,
        pages: u32,
        model_id: String,
        detent: OcrDetent,
        characters: usize,
        elapsed_ms: u64,
        /// True when the read stopped because it ran out of decode budget
        /// rather than because the model finished.
        ///
        /// It is the difference between "this is the page" and "this is as
        /// much as fitted", and a looping read produces the second while
        /// looking exactly like the first.
        hit_decode_cap: bool,
    },
}

/// Sends one already-stored image to the OCR model and returns what it read.
///
/// This is the path verified end to end against the real model, so it is
/// reused verbatim for both a directly attached image and a rendered page of
/// a scanned PDF rather than reimplemented for each.
///
/// `detent` is the slider stop the person chose. It used to be hard-coded to
/// `Detailed`, which made the chat path unable to trade accuracy for speed at
/// all — the slider existed only on the scan screen.
#[allow(clippy::too_many_arguments)]
async fn ocr_one_image(
    app: &AppHandle,
    registry: &ModelRegistry,
    servers: &ModelServers,
    image: &std::path::Path,
    detent: OcrDetent,
    // The attachment this page belongs to, for the events.
    name: &str,
    page: u32,
    pages: u32,
) -> Result<String, String> {
    let profile = detent.profile();
    let model_id = ocr_model_id(detent);
    let entry = registry
        .find(model_id)
        .ok_or_else(|| format!("{model_id} is not in the registry, so images cannot be read."))?
        .clone();

    let vram = crate::system_analyzer::gpu_collector::installed_gpus()
        .iter()
        .map(|gpu| gpu.dedicated_video_memory_bytes)
        .max()
        .unwrap_or(0);
    let plan = crate::ai_engine::vram_planner::plan_gpu_offload(
        vram,
        entry.weights_bytes,
        entry.context_length,
        None,
    );
    let endpoint = servers
        .endpoint_for(&entry, registry.models_dir(), &plan)
        .await
        .map_err(|e| e.to_string())?;

    crate::serving::probe::check_loopback(&endpoint.base_url).map_err(|outcome| {
        format!(
            "refusing to send an attachment off-machine: {}",
            outcome.explain(&endpoint.base_url)
        )
    })?;

    // arjun-egress-ok: loopback only, enforced by the check above.
    let client = reqwest::Client::new();
    let text = std::sync::Arc::new(std::sync::Mutex::new(String::new()));
    let sink = text.clone();
    let emitter = app.clone();
    let file = name.to_string();
    let started = std::time::Instant::now();
    let summary = stream_ocr(
        &client,
        &endpoint.base_url,
        &endpoint.served_model_id,
        image,
        &profile,
        Arc::new(AtomicBool::new(false)),
        move |event| match event {
            OcrEvent::Text { index, delta } => {
                if let Ok(mut t) = sink.lock() {
                    t.push_str(&delta);
                }
                let _ = emitter.emit(
                    "attachment:ocr",
                    AttachmentOcrEvent::Text {
                        name: file.clone(),
                        page,
                        index,
                        delta,
                    },
                );
            }
            OcrEvent::Region { index, label, .. } => {
                let _ = emitter.emit(
                    "attachment:ocr",
                    AttachmentOcrEvent::Region {
                        name: file.clone(),
                        page,
                        index,
                        label,
                    },
                );
            }
        },
    )
    .await
    .map_err(|e| format!("reading the page failed: {e:#}"))?;
    let read = text.lock().map(|t| t.clone()).unwrap_or_default();
    // `hit_decode_cap` used to be dropped on the floor here, and dropping it
    // is how a page that filled its entire token budget with one repeated
    // character reached the answer looking like an ordinary read. A page that
    // stopped because it ran out of budget did not finish, and the reader has
    // to be told which of the two happened.
    let _ = app.emit(
        "attachment:ocr",
        AttachmentOcrEvent::Page {
            name: name.to_string(),
            page,
            pages,
            model_id: model_id.to_string(),
            detent,
            characters: read.chars().count(),
            elapsed_ms: started.elapsed().as_millis() as u64,
            hit_decode_cap: summary.hit_decode_cap,
        },
    );
    Ok(read)
}

/// Runs the local one-shot extractor over a stored document.
///
/// A separate short-lived process rather than the long-lived document
/// sidecar: this needs no state between calls, and a crash in a PDF parser
/// then takes nothing else down with it.
fn run_extractor(path: &std::path::Path, out_dir: &std::path::Path) -> Result<Extracted, String> {
    let script = extractor_script().ok_or_else(|| {
        "the document extractor was not found next to the application".to_string()
    })?;
    let output = crate::system_analyzer::process_utils::create_hidden_command("python")
        .arg(&script)
        .arg(path)
        .arg(out_dir)
        .output()
        .map_err(|e| format!("the document extractor could not be started: {e}"))?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let line = stdout
        .lines()
        .rev()
        .find(|l| l.trim_start().starts_with('{'))
        .ok_or_else(|| {
            format!(
                "the document extractor returned nothing usable: {}",
                String::from_utf8_lossy(&output.stderr).trim()
            )
        })?;
    let parsed: Extracted = serde_json::from_str(line)
        .map_err(|e| format!("the document extractor returned unreadable output: {e}"))?;
    if let Some(error) = parsed.error {
        return Err(error);
    }
    Ok(parsed)
}

/// Decodes one attachment, stores it content-addressed, and turns it into
/// text a language model can reason about.
///
/// The routing is the design. A PDF that already carries a text layer is read
/// by parsing it, not by rendering pages and asking a vision model to read
/// back text the file was holding all along. Only images and genuine scans
/// reach the OCR model. The composer cannot influence this — the frontend can
/// only hand over bytes.
pub async fn read_attachment(
    app: &AppHandle,
    registry: &ModelRegistry,
    servers: &ModelServers,
    attachment: &ChatAttachment,
    detent: OcrDetent,
    tag: &StageTag,
) -> Result<AttachmentRead, String> {
    let bytes = base64::Engine::decode(
        &base64::engine::general_purpose::STANDARD,
        &attachment.data_base64,
    )
    .map_err(|e| format!("{} could not be decoded: {e}", attachment.name))?;
    let kind = validate_attachment(&attachment.name, &attachment.mime, bytes.len())?;

    let sha256 = {
        use sha2::{Digest, Sha256};
        let mut h = Sha256::new();
        h.update(&bytes);
        format!("{:x}", h.finalize())
    };

    let base = app
        .path()
        .app_data_dir()
        .map_err(|e| format!("no app data directory: {e}"))?
        .join("documents")
        .join("attachments")
        .join(&sha256);
    std::fs::create_dir_all(&base)
        .map_err(|e| format!("could not store {}: {e}", attachment.name))?;

    progress(app, tag, &attachment.name, "reading", None, None, None);

    // What the answer will say about how this file was read. Filled in by
    // whichever branch actually runs, so it reports the path taken rather
    // than the path expected.
    let read_kind: String;
    let mut read_pages: u32 = 1;
    let mut ocr_model: Option<String> = None;

    let text = match kind {
        AttachmentKind::Image(ext) => {
            let stored = base.join(format!("page-1.{ext}"));
            if !stored.exists() {
                std::fs::write(&stored, &bytes)
                    .map_err(|e| format!("could not store {}: {e}", attachment.name))?;
            }
            progress(
                app,
                tag,
                &attachment.name,
                "understanding",
                Some(1),
                Some(1),
                Some("image".into()),
            );
            read_kind = "image".into();
            ocr_model = Some(ocr_model_id(detent).to_string());
            ocr_one_image(
                app,
                registry,
                servers,
                &stored,
                detent,
                &attachment.name,
                1,
                1,
            )
            .await?
        }
        AttachmentKind::Document(ext) => {
            let stored = base.join(format!("source.{ext}"));
            if !stored.exists() {
                std::fs::write(&stored, &bytes)
                    .map_err(|e| format!("could not store {}: {e}", attachment.name))?;
            }
            progress(app, tag, &attachment.name, "preparing", None, None, None);
            let extracted = run_extractor(&stored, &base)?;
            let pages = extracted.pages.max(1);

            read_kind = extracted.kind.clone();
            read_pages = pages;

            if extracted.page_images.is_empty() {
                // A text layer, a spreadsheet, a Word file, a plain text file.
                // No model was needed to read it, and none was used.
                progress(
                    app,
                    tag,
                    &attachment.name,
                    "extracting",
                    None,
                    Some(pages),
                    Some(extracted.kind.clone()),
                );
                extracted.text
            } else {
                // A scan. Each rendered page goes to the OCR model, and the
                // counter the UI shows is this loop's real position.
                let mut parts = Vec::new();
                let total = extracted.page_images.len() as u32;
                read_pages = total;
                ocr_model = Some(ocr_model_id(detent).to_string());
                for (index, image) in extracted.page_images.iter().enumerate() {
                    progress(
                        app,
                        tag,
                        &attachment.name,
                        "understanding",
                        Some(index as u32 + 1),
                        Some(total),
                        Some(extracted.kind.clone()),
                    );
                    let page_text = ocr_one_image(
                        app,
                        registry,
                        servers,
                        image,
                        detent,
                        &attachment.name,
                        index as u32 + 1,
                        total,
                    )
                    .await?;
                    if !page_text.trim().is_empty() {
                        parts.push(format!("--- page {} ---\n{}", index + 1, page_text.trim()));
                    }
                }
                if extracted.truncated {
                    // Said plainly rather than letting the answer silently
                    // describe only part of the document.
                    parts.push(format!(
                        "(only the first {total} of {pages} pages were read)"
                    ));
                }
                parts.join("\n\n")
            }
        }
    };

    progress(app, tag, &attachment.name, "done", None, None, None);
    Ok(AttachmentRead {
        name: attachment.name.clone(),
        sha256,
        text: text.trim().to_string(),
        kind: read_kind,
        pages: read_pages,
        ocr_detent: ocr_model.as_ref().map(|_| detent),
        ocr_model_id: ocr_model,
    })
}

/// Which weight file a stop loads.
///
/// One place, so the routing explanation the person reads and the server the
/// request goes to cannot name different models.
pub const fn ocr_model_id(detent: OcrDetent) -> &'static str {
    match detent.profile().tier {
        OcrTier::High => "unlimited-ocr-q6-k",
        OcrTier::Fast => "unlimited-ocr-q4-k-m",
    }
}

/// Set while a page is being read, so a second call can stop it.
#[derive(Default)]
pub struct ScanCancel(pub Arc<AtomicBool>);

/// What the UI receives on `ocr:span`.
///
/// Carries the model's own numbers alongside the mapped ones. Keeping the raw
/// box means a future change to the emitted format shows up as a mismatch
/// that can be asserted on, instead of an overlay that quietly drifts.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase", tag = "event")]
enum SpanPayload {
    /// The inner `rename_all` is load-bearing and easy to lose: the container
    /// attribute renames *variants*, not their fields, so without this the UI
    /// receives `page_box` while `ocr.service.ts` reads `pageBox` — and the
    /// overlay silently never draws.
    #[serde(rename_all = "camelCase")]
    Region {
        index: usize,
        label: String,
        bbox: Option<RawBox>,
        page_box: Option<PageBox>,
    },
    #[serde(rename_all = "camelCase")]
    Text { index: Option<usize>, delta: String },
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct StatusPayload {
    page: u32,
    state: &'static str,
    tokens: u32,
    elapsed_ms: u64,
    /// Only ever a measured figure. Absent rather than estimated.
    tokens_per_second: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct ErrorPayload {
    page: u32,
    reason: String,
}

/// Where a rasterised page lives.
///
/// Rendering a PDF to page images happens upstream of this command; it is not
/// done here. A missing file is reported as exactly that rather than being
/// silently treated as an empty page.
fn page_image_path(app: &AppHandle, sha256: &str, page: u32) -> Result<PathBuf, String> {
    let base = app
        .path()
        .app_data_dir()
        .map_err(|e| format!("no app data directory: {e}"))?;
    let path = base
        .join("documents")
        .join("pages")
        .join(sha256)
        .join(format!("page-{page}.png"));
    if !path.exists() {
        return Err(format!(
            "page {page} of {sha256} has not been rendered to an image yet ({}). \
             Rasterise the document before reading it.",
            path.display()
        ));
    }
    Ok(path)
}

/// Reads one page, streaming regions to the UI as the model finds them.
#[tauri::command]
pub async fn scan_page(
    app: AppHandle,
    registry: State<'_, Arc<ModelRegistry>>,
    servers: State<'_, Arc<ModelServers>>,
    cancel: State<'_, ScanCancel>,
    document_sha256: String,
    page: u32,
    detent: OcrDetent,
) -> Result<(), String> {
    let flag = cancel.0.clone();
    // Cleared here rather than at the end of the previous run: a run that
    // failed or was dropped must not leave the next one pre-cancelled.
    flag.store(false, Ordering::Relaxed);

    let profile = detent.profile();
    let model_id = match profile.tier {
        OcrTier::High => "unlimited-ocr-q6-k",
        OcrTier::Fast => "unlimited-ocr-q4-k-m",
    };

    let image = page_image_path(&app, &document_sha256, page)?;
    let (page_w, page_h) = png_dimensions(&image)?;
    // Under the measured convention the model normalises against its own
    // input, so the page's own size is both the source and the target space.
    let geometry = PageGeometry {
        page_width: page_w,
        page_height: page_h,
        input_width: page_w,
        input_height: page_h,
    };

    let entry = registry
        .find(model_id)
        .ok_or_else(|| {
            format!("{model_id} is not in the registry. Merge config/ocr-model-registry.json.")
        })?
        .clone();

    let vram = crate::system_analyzer::gpu_collector::installed_gpus()
        .iter()
        .map(|gpu| gpu.dedicated_video_memory_bytes)
        .max()
        .unwrap_or(0);
    let plan = crate::ai_engine::vram_planner::plan_gpu_offload(
        vram,
        entry.weights_bytes,
        entry.context_length,
        None,
    );

    let endpoint = servers
        .endpoint_for(&entry, registry.models_dir(), &plan)
        .await
        .map_err(|e| e.to_string())?;

    let _ = app.emit(
        "ocr:status",
        StatusPayload {
            page,
            state: "reading",
            tokens: 0,
            elapsed_ms: 0,
            tokens_per_second: None,
        },
    );

    // Enforced, not assumed. A managed endpoint is always loopback, but an
    // operator can point a registry entry at an external server, and this
    // command must not become the one place a document leaves the machine.
    // The annotation below is only honest because of this check.
    crate::serving::probe::check_loopback(&endpoint.base_url).map_err(|outcome| {
        format!(
            "refusing to send a document off-machine: {}",
            outcome.explain(&endpoint.base_url)
        )
    })?;

    // arjun-egress-ok: loopback only. The check above rejects any
    // non-loopback base URL, so the only host this client can address is the
    // local llama.cpp server ARJUN itself started. Sovereignty: no remote.
    let client = reqwest::Client::new();
    let emitter = app.clone();
    let result = stream_ocr(
        &client,
        &endpoint.base_url,
        &endpoint.served_model_id,
        &image,
        &profile,
        flag.clone(),
        move |event| {
            let payload = match event {
                OcrEvent::Region { index, label, bbox } => SpanPayload::Region {
                    index,
                    label,
                    bbox,
                    // Mapped only when the convention has actually been
                    // measured; otherwise the UI draws no overlay rather
                    // than a plausible-looking wrong one.
                    page_box: bbox.and_then(|raw| {
                        CALIBRATED_COORD_SPACE.map(|space| to_page(raw, space, geometry))
                    }),
                },
                OcrEvent::Text { index, delta } => SpanPayload::Text { index, delta },
            };
            let _ = emitter.emit("ocr:span", payload);
        },
    )
    .await;

    match result {
        Ok(summary) => {
            let seconds = summary.elapsed_ms as f64 / 1000.0;
            let _ = app.emit(
                "ocr:status",
                StatusPayload {
                    page,
                    state: if summary.cancelled { "failed" } else { "done" },
                    tokens: summary.tokens,
                    elapsed_ms: summary.elapsed_ms,
                    // Measured, or absent. Never a plausible constant.
                    tokens_per_second: if seconds > 0.0 && summary.tokens > 0 {
                        Some(summary.tokens as f64 / seconds)
                    } else {
                        None
                    },
                },
            );
            if summary.hit_decode_cap {
                // The page stopped because it ran out of budget. On the Fast
                // tier that is the signature of the loop the DRY substitute
                // is supposed to prevent, so it is surfaced, not hidden.
                let _ = app.emit(
                    "ocr:error",
                    ErrorPayload {
                        page,
                        reason: "Generation stopped at the decode limit rather than \
                                 finishing. On the Fast tier this usually means the \
                                 repetition guard did not hold; try a higher stop."
                            .to_string(),
                    },
                );
            }
            Ok(())
        }
        Err(error) => {
            let reason = format!("{error:#}");
            let _ = app.emit(
                "ocr:status",
                StatusPayload {
                    page,
                    state: "failed",
                    tokens: 0,
                    elapsed_ms: 0,
                    tokens_per_second: None,
                },
            );
            let _ = app.emit(
                "ocr:error",
                ErrorPayload {
                    page,
                    reason: reason.clone(),
                },
            );
            Err(reason)
        }
    }
}

/// Stops the page currently being read. Safe to call when nothing is running.
#[tauri::command]
pub fn cancel_scan(cancel: State<'_, ScanCancel>) {
    cancel.0.store(true, Ordering::Relaxed);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A photograph has no text layer, so a model has to look at it. This is
    /// the case the composer's hint exists for.
    #[test]
    fn an_image_is_planned_for_the_ocr_model() {
        let plan = plan_attachment("scan.png", "image/png");
        assert_eq!(plan.route, "image");
        assert!(plan.needs_ocr);
        assert!(plan.refusal.is_none());
        assert!(
            plan.explanation.contains("document-OCR model"),
            "the hint has to name what will read it: {}",
            plan.explanation
        );
    }

    /// The whole point of routing by type: a spreadsheet already carries its
    /// text, and claiming an OCR model will read it would be a claim about
    /// work that never happens.
    #[test]
    fn a_spreadsheet_is_planned_without_any_model() {
        let plan = plan_attachment("readings.xlsx", "");
        assert_eq!(plan.route, "document");
        assert!(!plan.needs_ocr);
        assert!(
            plan.explanation.contains("no model is needed"),
            "it must say no model is needed: {}",
            plan.explanation
        );
    }

    /// A PDF is the honest "maybe": whether it is a scan is only known once
    /// the extractor opens it, and the wording says so rather than promising
    /// one path.
    #[test]
    fn a_pdf_is_planned_as_possibly_needing_ocr() {
        let plan = plan_attachment("drawing.pdf", "application/pdf");
        assert_eq!(plan.route, "document");
        assert!(plan.needs_ocr);
        assert!(plan.explanation.contains("text layer"));
    }

    /// The plan and the run must refuse the same files. A plan that accepts
    /// what the run rejects is how a person gets a hint and then an error.
    #[test]
    fn an_unreadable_file_is_refused_with_the_same_sentence_the_run_uses() {
        let plan = plan_attachment("weird.bin", "application/octet-stream");
        assert_eq!(plan.route, "rejected");
        assert!(!plan.needs_ocr);
        let refusal = plan.refusal.expect("a rejected file carries its reason");
        let from_run = validate_attachment("weird.bin", "application/octet-stream", 1)
            .expect_err("the run rejects it too");
        assert_eq!(refusal, from_run);
    }

    /// The slider stop and the weight file are one decision. If these ever
    /// disagree, the routing explanation names a model the request did not
    /// go to.
    #[test]
    fn each_stop_names_the_weight_file_its_profile_asks_for() {
        assert_eq!(ocr_model_id(OcrDetent::Fastest), "unlimited-ocr-q4-k-m");
        assert_eq!(ocr_model_id(OcrDetent::Fast), "unlimited-ocr-q4-k-m");
        assert_eq!(ocr_model_id(OcrDetent::Detailed), "unlimited-ocr-q6-k");
        assert_eq!(ocr_model_id(OcrDetent::Maximum), "unlimited-ocr-q6-k");
    }

    #[test]
    fn the_measured_coordinate_space_maps_a_known_box_onto_the_page() {
        // Real numbers from the calibration run: the page's bottom-right
        // marker has ink at y=1299 of 1400 and the model reported y=923.
        // If the convention were ever mis-set to InputPixels this lands at
        // 923 instead of ~1294 and the assertion fails.
        let space = CALIBRATED_COORD_SPACE.expect("calibrated by the Phase 0 gate");
        assert_eq!(space, CoordSpace::Normalised);
        let geometry = crate::ai_engine::ocr_profile::PageGeometry {
            page_width: 1000,
            page_height: 1400,
            input_width: 1000,
            input_height: 1400,
        };
        let mapped = to_page(
            RawBox {
                x1: 615,
                y1: 923,
                x2: 866,
                y2: 956,
            },
            space,
            geometry,
        );
        assert!(mapped.in_bounds, "the marker must land on the page");
        assert!(
            (mapped.y1 - 1294).abs() <= 12,
            "expected ~1294 for the footer marker, got {}",
            mapped.y1
        );
        assert!((mapped.x1 - 616).abs() <= 12, "got x1={}", mapped.x1);
    }

    #[test]
    fn a_span_payload_matches_the_typescript_contract() {
        let region = SpanPayload::Region {
            index: 0,
            label: "title".into(),
            bbox: None,
            page_box: None,
        };
        let json = serde_json::to_string(&region).expect("serialises");
        assert!(json.contains(r#""event":"region""#), "got {json}");
        assert!(json.contains(r#""pageBox""#), "got {json}");
    }

    #[test]
    fn cancelling_is_sticky_until_the_next_scan_clears_it() {
        let flag = ScanCancel::default();
        assert!(!flag.0.load(Ordering::Relaxed));
        flag.0.store(true, Ordering::Relaxed);
        assert!(flag.0.load(Ordering::Relaxed));
    }

    #[test]
    fn an_attachment_of_an_unreadable_type_is_refused_by_name() {
        let err = validate_attachment("clip.mp4", "video/mp4", 10).unwrap_err();
        assert!(
            err.contains("clip.mp4"),
            "the person must see which file: {err}"
        );
        assert!(err.contains("PDF"), "and what would work: {err}");
    }

    #[test]
    fn an_empty_attachment_is_refused_rather_than_read_as_a_blank_page() {
        assert!(validate_attachment("scan.png", "image/png", 0)
            .unwrap_err()
            .contains("empty"));
    }

    #[test]
    fn an_oversized_attachment_is_refused_before_it_reaches_the_model() {
        // A refusal the person can read beats a request that dies inside
        // llama-server with no explanation.
        let err =
            validate_attachment("huge.png", "image/png", MAX_ATTACHMENT_BYTES + 1).unwrap_err();
        assert!(err.contains("limit"), "got {err}");
    }

    #[test]
    fn the_readable_image_types_map_to_the_extensions_the_model_accepts() {
        assert_eq!(
            validate_attachment("a.png", "image/png", 9).unwrap(),
            AttachmentKind::Image("png")
        );
        assert_eq!(
            validate_attachment("a.jpg", "image/jpeg", 9).unwrap(),
            AttachmentKind::Image("jpg")
        );
        assert_eq!(
            validate_attachment("a.webp", "image/webp", 9).unwrap(),
            AttachmentKind::Image("webp")
        );
    }

    /// The regression for the PDF refusal: a PDF used to be rejected because
    /// the only accepted types were the three the vision model reads. It must
    /// now route to the extractor, not to OCR.
    #[test]
    fn a_pdf_routes_to_the_document_extractor_not_to_ocr() {
        assert_eq!(
            validate_attachment("report.pdf", "application/pdf", 4096).unwrap(),
            AttachmentKind::Document("pdf")
        );
    }

    #[test]
    fn text_native_formats_route_to_the_extractor_so_no_model_reads_them() {
        for (name, mime) in [
            ("notes.txt", "text/plain"),
            ("readme.md", "text/markdown"),
            ("rows.csv", "text/csv"),
            ("config.json", "application/json"),
        ] {
            match validate_attachment(name, mime, 64) {
                Ok(AttachmentKind::Document(_)) => {}
                other => panic!("{name} should be a document, got {other:?}"),
            }
        }
    }

    #[test]
    fn office_formats_route_to_the_extractor() {
        assert_eq!(
            validate_attachment(
                "a.docx",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                64
            )
            .unwrap(),
            AttachmentKind::Document("docx")
        );
        assert_eq!(
            validate_attachment(
                "a.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                64
            )
            .unwrap(),
            AttachmentKind::Document("xlsx")
        );
    }

    #[test]
    fn an_image_still_routes_to_ocr_so_the_verified_path_is_untouched() {
        // Guards the working Unlimited-OCR path against a routing change.
        assert!(matches!(
            validate_attachment("scan.png", "image/png", 4096).unwrap(),
            AttachmentKind::Image(_)
        ));
    }

    #[test]
    fn a_browser_that_reports_no_mime_still_routes_by_extension() {
        // Some picks arrive with an empty `type`; the suffix is the fallback.
        assert_eq!(
            validate_attachment("report.pdf", "", 4096).unwrap(),
            AttachmentKind::Document("pdf")
        );
    }

    #[test]
    fn the_ui_receives_four_stops_in_slider_order() {
        let stops = detent_info();
        assert_eq!(stops.len(), 4);
        assert_eq!(stops[0].detent, OcrDetent::Fastest);
        assert_eq!(stops[3].detent, OcrDetent::Maximum);
    }

    #[test]
    fn every_stop_reports_the_numbers_its_profile_will_actually_use() {
        // The whole reason this command exists rather than a hardcoded table
        // in the frontend.
        for stop in detent_info() {
            let profile = stop.detent.profile();
            assert_eq!(stop.max_image_tokens, profile.max_image_tokens);
            assert_eq!(stop.max_decode_tokens, profile.max_decode_tokens);
            assert_eq!(stop.tier, profile.tier);
        }
    }

    #[test]
    fn the_payload_is_camel_case_for_the_typescript_side() {
        // `src/services/ocr.service.ts` declares maxImageTokens / tierLabel;
        // a rename here would break the slider silently at runtime.
        let json = serde_json::to_string(&detent_info()[0]).expect("serialises");
        assert!(json.contains("\"maxImageTokens\""), "got {json}");
        assert!(json.contains("\"tierLabel\""), "got {json}");
        assert!(json.contains("\"maxDecodeTokens\""), "got {json}");
        assert!(
            json.contains("\"fastest\""),
            "detent must serialise camelCase: {json}"
        );
    }
}
