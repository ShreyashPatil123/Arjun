//! Axum server: routes, handlers, and SSE streaming for both protocols.

use std::convert::Infallible;
use std::sync::{Arc, Mutex};

use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
// Only futures_util's StreamExt is imported: tokio_stream's version defines the
// same combinators and importing both makes every call ambiguous.
use futures_util::stream::{self, Stream, StreamExt};
use tokio_stream::wrappers::UnboundedReceiverStream;

use crate::ai_engine::scheduler::{
    CancelOnDrop, Canceller, GenerationHandle, GenerationJob, JobOrigin,
};
use crate::ai_engine::traits::{ChatMessage, GenerationParams, StreamChunk};
use crate::gateway::anthropic::{self, MessagesRequest, MessagesResponse, StreamEvents};
use crate::gateway::guard::origin_guard;
use crate::gateway::openai::{ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, ModelList};
use crate::gateway::state::{client_label, GatewayState};

/// Running server; call [`GatewayHandle::stop`] to shut it down.
///
/// **The handle must be kept alive for as long as the server should run.**
/// Dropping it drops the shutdown sender, which resolves the graceful-shutdown
/// future and stops the server accepting connections — within milliseconds of
/// it reporting that it is listening. Store it in application state rather than
/// letting it fall out of scope.
#[derive(Debug)]
pub struct GatewayHandle {
    pub port: u16,
    shutdown: Option<tokio::sync::oneshot::Sender<()>>,
}

impl GatewayHandle {
    pub fn stop(mut self) {
        if let Some(tx) = self.shutdown.take() {
            let _ = tx.send(());
        }
    }
}

/// Binds the gateway and serves until shutdown.
///
/// A bind failure (usually the port being in use) is returned rather than
/// panicking, so the desktop app still opens and the user can choose another port.
/// Binds `addr`, retrying briefly while it is still held.
///
/// A port busy at startup is nearly always this app's own previous instance
/// finishing its shutdown — restarting Sarathi is exactly when this happens, and
/// the socket clears in well under a second. Retrying keeps the address the user
/// configured, which matters because anything pointed at it by hand would
/// otherwise be left behind by a fallback.
async fn bind_with_retry(
    addr: std::net::SocketAddr,
) -> std::io::Result<tokio::net::TcpListener> {
    const GRACE: std::time::Duration = std::time::Duration::from_secs(3);
    const POLL: std::time::Duration = std::time::Duration::from_millis(150);

    let deadline = std::time::Instant::now() + GRACE;
    loop {
        match tokio::net::TcpListener::bind(addr).await {
            Ok(listener) => return Ok(listener),
            Err(e) if std::time::Instant::now() < deadline => {
                log::debug!("[GATEWAY] {addr} busy ({e}); retrying");
                tokio::time::sleep(POLL).await;
            }
            Err(e) => return Err(e),
        }
    }
}

pub async fn start_gateway(state: Arc<GatewayState>) -> anyhow::Result<GatewayHandle> {
    let requested_port = state.port();
    let app = router(state.clone());

    // Loopback only. Never 0.0.0.0 — that would expose the model to the network.
    let addr = std::net::SocketAddr::from(([127, 0, 0, 1], requested_port));
    let listener = match bind_with_retry(addr).await {
        Ok(listener) => listener,
        Err(first) => {
            // Binding once and giving up left the gateway down for the entire
            // session whenever the port was momentarily busy — most often this
            // app's own previous instance still releasing its socket after a
            // quick restart. Every client then got ConnectionRefused with
            // nothing to explain it, and no amount of retrying on their side
            // could recover, because nothing was ever going to listen.
            //
            // An OS-assigned port always succeeds. The bound port is read back
            // below and published to state, and tools are handed that at launch
            // rather than the configured value, so a fallback stays invisible.
            log::warn!(
                "[GATEWAY] Could not bind 127.0.0.1:{requested_port} ({first}); falling back to a free port"
            );
            let any_port = std::net::SocketAddr::from(([127, 0, 0, 1], 0));
            tokio::net::TcpListener::bind(any_port).await.map_err(|e| {
                anyhow::anyhow!(
                    "could not bind gateway to 127.0.0.1:{requested_port} ({first}), \
                     nor to any free port ({e})"
                )
            })?
        }
    };

    // Read the port back from the socket rather than trusting the request.
    // Port 0 means "any free port", and the OS picks the real one — reporting
    // the requested value would hand callers a port nothing is listening on,
    // and log a connection string that cannot work.
    let port = listener
        .local_addr()
        .map_err(|e| anyhow::anyhow!("bound gateway but could not read its address: {e}"))?
        .port();

    if port != requested_port {
        log::warn!(
            "[GATEWAY] Configured port {requested_port} was unavailable; serving on {port} instead"
        );
    }
    // Published so the dashboard and every launched tool use the live address.
    state.set_port(port);

    let (tx, rx) = tokio::sync::oneshot::channel::<()>();

    tokio::spawn(async move {
        let server = axum::serve(listener, app).with_graceful_shutdown(async {
            let _ = rx.await;
        });
        if let Err(e) = server.await {
            log::error!("[GATEWAY] Server stopped with error: {e}");
        }
    });

    log::info!("[GATEWAY] Listening on http://127.0.0.1:{port}");
    Ok(GatewayHandle { port, shutdown: Some(tx) })
}

pub fn router(state: Arc<GatewayState>) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(openai_chat))
        .route("/v1/messages", post(anthropic_messages))
        .layer(axum::middleware::from_fn(origin_guard))
        .with_state(state)
}

// ─── Simple endpoints ───────────────────────────────────────────────────────

async fn health(State(state): State<Arc<GatewayState>>) -> impl IntoResponse {
    let model = state.inference.get_loaded_model_info();
    Json(serde_json::json!({
        "status": "ok",
        "modelLoaded": model.is_some(),
        "model": model.map(|m| m.model_id),
    }))
}

async fn list_models(State(state): State<Arc<GatewayState>>) -> impl IntoResponse {
    let id = state.inference.get_loaded_model_info().map(|m| m.model_id);
    Json(ModelList::single(id, now_secs()))
}

// ─── Helpers ────────────────────────────────────────────────────────────────

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn now_millis() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn short_id(prefix: &str) -> String {
    format!("{prefix}{}", uuid::Uuid::new_v4().simple())
}

fn json_of<T: serde::Serialize>(value: &T) -> String {
    serde_json::to_string(value).unwrap_or_default()
}

/// The model being served, or a 503 explaining what to do about it.
///
/// A clear message here is what stops users debugging their client config for an
/// hour when the real problem is that no model is loaded.
fn active_model(state: &GatewayState) -> Result<String, Response> {
    match state.inference.get_loaded_model_info() {
        Some(info) => Ok(info.model_id),
        None => Err((
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({
                "error": {
                    "type": "no_model_loaded",
                    "message": "No model is loaded. Open Sarathi and load a model, then retry."
                }
            })),
        )
            .into_response()),
    }
}

fn submit(
    state: &GatewayState,
    messages: Vec<ChatMessage>,
    params: GenerationParams,
    client: &str,
) -> Result<GenerationHandle, Response> {
    state
        .scheduler
        .submit(GenerationJob {
            messages,
            params,
            // Capability routing is opt-in for gateway traffic: external tools
            // send their own system prompts and sampling, and overriding those
            // silently can break output they depend on.
            capability: None,
            origin: JobOrigin::Gateway { client: client.to_string() },
        })
        .map_err(|e| {
            (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({
                    "error": { "type": "scheduler_unavailable", "message": e.to_string() }
                })),
            )
                .into_response()
        })
}

/// Drains the whole answer for non-streaming requests.
async fn collect(handle: &mut GenerationHandle) -> (String, String, u32) {
    let mut text = String::new();
    let mut finish = "stop".to_string();
    let mut tokens = 0u32;

    while let Some(chunk) = handle.chunks.recv().await {
        text.push_str(&chunk.text);
        if let Some(n) = chunk.tokens_generated {
            tokens = n;
        }
        if chunk.is_final {
            if let Some(reason) = chunk.finish_reason {
                finish = reason;
            }
            break;
        }
    }
    (text, finish, tokens)
}

/// How a generation ended, shared between the body and tail of a stream.
#[derive(Clone)]
struct Outcome(Arc<Mutex<(String, u32)>>);

impl Outcome {
    fn new() -> Self {
        Self(Arc::new(Mutex::new(("stop".to_string(), 0))))
    }
    fn record(&self, reason: String, tokens: u32) {
        if let Ok(mut g) = self.0.lock() {
            *g = (reason, tokens);
        }
    }
    fn get(&self) -> (String, u32) {
        self.0.lock().map(|g| g.clone()).unwrap_or_else(|_| ("stop".to_string(), 0))
    }
}

// ─── OpenAI ─────────────────────────────────────────────────────────────────

async fn openai_chat(
    State(state): State<Arc<GatewayState>>,
    headers: HeaderMap,
    Json(req): Json<ChatCompletionRequest>,
) -> Response {
    let client = client_label(headers.get("user-agent").and_then(|v| v.to_str().ok()));
    state.record_request(&client, "openai", now_millis());

    let model = match active_model(&state) {
        Ok(m) => m,
        Err(resp) => {
            state.finish_request();
            return resp;
        }
    };

    let mut handle = match submit(&state, req.to_chat_messages(), req.to_generation_params(), &client) {
        Ok(h) => h,
        Err(resp) => {
            state.finish_request();
            return resp;
        }
    };

    let id = short_id("chatcmpl-");
    let created = now_secs();

    if !req.stream {
        let (text, finish, tokens) = collect(&mut handle).await;
        state.finish_request();
        return Json(ChatCompletionResponse::new(id, created, model, text, &finish, tokens)).into_response();
    }

    Sse::new(openai_stream(state.clone(), handle, id, created, model))
        .keep_alive(KeepAlive::default())
        .into_response()
}

/// `opening chunk` → `text chunk`* → `closing chunk` → `[DONE]`.
///
/// All parts share one id, which clients use to correlate the stream.
fn openai_stream(
    state: Arc<GatewayState>,
    handle: GenerationHandle,
    id: String,
    created: u64,
    model: String,
) -> impl Stream<Item = Result<Event, Infallible>> {
    let canceller: Canceller = handle.canceller();
    // Held by the body stream, so it is dropped whenever the response stream is
    // — including when the client hangs up mid-prefill, which the tail below
    // never sees because that only runs on a fully read answer.
    let mut guard = CancelOnDrop::new(canceller.clone());
    let chunks = UnboundedReceiverStream::new(handle.chunks);

    let (head_id, head_model) = (id.clone(), model.clone());
    let head = stream::once(async move {
        Ok(Event::default().data(json_of(&ChatCompletionChunk::opening(&head_id, created, &head_model))))
    });

    let (body_id, body_model) = (id.clone(), model.clone());
    let body = chunks.filter_map(move |chunk: StreamChunk| {
        if chunk.is_final {
            guard.disarm();
        }
        let event = if chunk.is_final {
            let reason = chunk.finish_reason.clone().unwrap_or_else(|| "stop".to_string());
            Some(Event::default().data(json_of(&ChatCompletionChunk::closing(
                &body_id, created, &body_model, &reason,
            ))))
        } else if chunk.text.is_empty() {
            None
        } else {
            Some(Event::default().data(json_of(&ChatCompletionChunk::text(
                &body_id, created, &body_model, chunk.text,
            ))))
        };
        async move { event.map(Ok) }
    });

    let tail = stream::once(async move {
        // The client has read everything; make sure nothing keeps generating.
        canceller.cancel();
        state.finish_request();
        Ok(Event::default().data("[DONE]"))
    });

    head.chain(body).chain(tail)
}

// ─── Anthropic ──────────────────────────────────────────────────────────────

async fn anthropic_messages(
    State(state): State<Arc<GatewayState>>,
    headers: HeaderMap,
    Json(req): Json<MessagesRequest>,
) -> Response {
    let client = client_label(headers.get("user-agent").and_then(|v| v.to_str().ok()));
    state.record_request(&client, "anthropic", now_millis());

    let model = match active_model(&state) {
        Ok(m) => m,
        Err(resp) => {
            state.finish_request();
            return resp;
        }
    };

    let mut handle = match submit(&state, req.to_chat_messages(), req.to_generation_params(), &client) {
        Ok(h) => h,
        Err(resp) => {
            state.finish_request();
            return resp;
        }
    };

    let id = short_id("msg_");

    if !req.stream {
        let (text, finish, tokens) = collect(&mut handle).await;
        state.finish_request();
        let stop = anthropic::map_stop_reason(&finish);
        return Json(MessagesResponse::new(id, model, text, stop, tokens)).into_response();
    }

    Sse::new(anthropic_stream(state.clone(), handle, id, model))
        .keep_alive(KeepAlive::default())
        .into_response()
}

/// The exact event order Anthropic clients require:
/// `message_start` → `content_block_start` → `content_block_delta`* →
/// `content_block_stop` → `message_delta` → `message_stop`.
fn anthropic_stream(
    state: Arc<GatewayState>,
    handle: GenerationHandle,
    id: String,
    model: String,
) -> impl Stream<Item = Result<Event, Infallible>> {
    let canceller: Canceller = handle.canceller();
    // See the OpenAI stream: the tail only runs for an answer the client read to
    // the end, so normal completion is not the case that needs covering.
    let mut guard = CancelOnDrop::new(canceller.clone());
    let chunks = UnboundedReceiverStream::new(handle.chunks);
    let outcome = Outcome::new();

    fn sse((name, data): (&'static str, String)) -> Result<Event, Infallible> {
        Ok(Event::default().event(name).data(data))
    }

    let head = stream::iter(vec![
        sse(StreamEvents::message_start(&id, &model)),
        sse(StreamEvents::content_block_start()),
    ]);

    let body_outcome = outcome.clone();
    let body = chunks.filter_map(move |chunk: StreamChunk| {
        let event = if chunk.is_final {
            // Generation ended on its own, so dropping the guard must not read
            // as an abandonment.
            guard.disarm();
            // Terminal events belong to `tail`; just record how it ended.
            let reason = anthropic::map_stop_reason(chunk.finish_reason.as_deref().unwrap_or("stop"));
            body_outcome.record(reason.to_string(), chunk.tokens_generated.unwrap_or(0));
            None
        } else if chunk.text.is_empty() {
            None
        } else {
            Some(sse(StreamEvents::content_block_delta(&chunk.text)))
        };
        async move { event }
    });

    let tail_outcome = outcome.clone();
    let tail = stream::once(async move {
        canceller.cancel();
        state.finish_request();
        tail_outcome.get()
    })
    .flat_map(|(stop_reason, tokens)| {
        stream::iter(vec![
            sse(StreamEvents::content_block_stop()),
            sse(StreamEvents::message_delta(&stop_reason, tokens)),
            sse(StreamEvents::message_stop()),
        ])
    });

    head.chain(body).chain(tail)
}

#[cfg(test)]
mod server_tests {
    use super::*;
    use crate::ai_engine::scheduler::GenerationScheduler;
    use crate::gateway::state::GatewayConfig;

    fn test_state(port: u16) -> Arc<GatewayState> {
        let inference = Arc::new(crate::ai_engine::manager::InferenceManager::new());
        let scheduler = Arc::new(GenerationScheduler::start(inference.clone()));
        Arc::new(GatewayState::new(
            scheduler,
            inference,
            GatewayConfig { enabled: true, port, apply_capabilities: false },
        ))
    }

    /// Port 0 asks the OS for a free port, so tests never collide with a running
    /// app or with each other.
    async fn start_on_ephemeral_port() -> (GatewayHandle, u16) {
        let handle = start_gateway(test_state(0)).await.expect("gateway should bind");
        let port = handle.port;
        assert_ne!(port, 0, "handle must report the OS-assigned port, not the requested 0");
        (handle, port)
    }

    #[tokio::test]
    async fn the_server_answers_while_the_handle_is_held() {
        let (_handle, port) = start_on_ephemeral_port().await;

        let body = reqwest::get(format!("http://127.0.0.1:{port}/health"))
            .await
            .expect("health should respond")
            .text()
            .await
            .expect("health should have a body");

        assert!(body.contains("\"status\""), "unexpected health body: {body}");
    }

    /// Regression: the handle owns the graceful-shutdown sender, so letting it
    /// fall out of scope stopped the server milliseconds after it logged
    /// "Listening". The app reported a healthy gateway while refusing every
    /// connection, which is exactly what happened in practice.
    #[tokio::test]
    async fn dropping_the_handle_stops_the_server() {
        let (handle, port) = start_on_ephemeral_port().await;
        let url = format!("http://127.0.0.1:{port}/health");

        assert!(
            reqwest::get(&url).await.is_ok(),
            "server should answer while the handle is alive"
        );

        drop(handle);
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;

        assert!(
            reqwest::get(&url).await.is_err(),
            "server should stop once the handle drops — if this ever fails, the \
             shutdown wiring changed and lib.rs may no longer need to retain it"
        );
    }

    #[tokio::test]
    async fn a_taken_port_falls_back_instead_of_leaving_the_gateway_down() {
        // Previously this returned an error and the app ran with no gateway for
        // the rest of the session: every client got ConnectionRefused, retrying
        // could never help because nothing would ever listen, and the dashboard
        // still reported the server as running. Serving somewhere is strictly
        // better than serving nowhere, and callers are handed the live port
        // rather than the configured one.
        let (_first, port) = start_on_ephemeral_port().await;

        let state = test_state(port);
        let second = start_gateway(state.clone())
            .await
            .expect("a taken port must not stop the gateway from starting");

        assert_ne!(second.port, port, "it must move off the port already in use");
        assert_ne!(second.port, 0, "the fallback must report a real bound port");

        // The live port has to be published, or launched tools are handed an
        // address nothing is listening on.
        assert_eq!(
            state.port(),
            second.port,
            "state must carry the port that was actually bound"
        );

        // And it must genuinely be accepting connections there.
        assert!(
            tokio::net::TcpStream::connect(("127.0.0.1", second.port))
                .await
                .is_ok(),
            "the fallback port must be live"
        );
    }
}
