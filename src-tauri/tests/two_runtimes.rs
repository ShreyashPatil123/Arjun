//! Both worlds, one router: the Phase 2 acceptance test.
//!
//! PS 26117 asks for *"model auto selection across at least two different task
//! types"*, and ARJUN's answer has to hold across two different **inference
//! runtimes** as well — because the models this problem needs live in two
//! ecosystems and neither covers it alone. GGUF served by `llama-server` is
//! where the good quantised coding and reasoning models are; Python served by
//! vLLM is where nearly every document-vision and OCR model is released.
//!
//! So this proves three things together:
//!
//! 1. a coding prompt and a document prompt route to **different models**;
//! 2. those models are on **different runtimes**;
//! 3. each resolves to a reachable OpenAI-compatible endpoint on loopback.
//!
//! ## What is real and what is not
//!
//! The vLLM endpoint is a real HTTP server on a real loopback port, so the
//! external-serving path — probe, readiness, endpoint resolution — runs exactly
//! as it does in production. It simply answers with a fixed model list instead
//! of loading 16 GB of weights.
//!
//! The llama.cpp path is exercised up to the launch command and no further,
//! because starting a real `llama-server` needs a binary and a GGUF that a test
//! machine has no business requiring. What that command contains is asserted
//! precisely, since that is where the mistakes live: a wrong `--host` would put
//! an inference server on the plant network.

use std::net::SocketAddr;
use std::path::{Path, PathBuf};

use sarathi_lib::ai_engine::vram_planner::plan_gpu_offload;
use sarathi_lib::registry::router::ModelRouter;
use sarathi_lib::registry::{ModelRegistry, ModelRole, Runtime};
use sarathi_lib::serving::{plan_launch, ModelServers, ServingSpec};

/// A registry holding one GGUF coding model and one Python document model.
///
/// Written to a real `registry.json` rather than constructed in memory, so the
/// manifest schema — the thing an administrator actually edits to add a model —
/// is what the test exercises.
fn registry_with_both_runtimes(dir: &Path, vllm_base_url: &str) -> ModelRegistry {
    let manifest = serde_json::json!({
        "models": [
            {
                "id": "qwen2.5-coder-7b",
                "name": "Qwen2.5 Coder 7B",
                "version": "1.0",
                "license": "Apache-2.0",
                "runtime": "llamaCpp",
                "roles": ["coding"],
                "quantization": "Q4_K_M",
                "parametersB": 7.0,
                "contextLength": 32768,
                "weightsBytes": 4_700_000_000u64,
                "permittedClassifications": ["internal", "processDiagram"],
                "path": "qwen2.5-coder-7b-q4.gguf",
                "enabled": true
            },
            {
                "id": "granite-docling-258m",
                "name": "Granite Docling 258M",
                "version": "1.0",
                "license": "Apache-2.0",
                "runtime": "pythonSidecar",
                "roles": ["documentOcr", "vision"],
                "parametersB": 0.258,
                "contextLength": 8192,
                "weightsBytes": 520_000_000u64,
                "permittedClassifications": ["internal", "processDiagram"],
                "path": "granite-docling-258m",
                "serving": { "mode": "external", "baseUrl": vllm_base_url },
                "enabled": true
            }
        ]
    });

    std::fs::write(
        dir.join("registry.json"),
        serde_json::to_string_pretty(&manifest).expect("manifest serialises"),
    )
    .expect("manifest is written");

    ModelRegistry::load(dir).expect("the manifest loads")
}

/// A stand-in for an operator's vLLM, answering `/v1/models` and nothing else.
async fn fake_vllm(models: Vec<&'static str>) -> (String, tokio::task::JoinHandle<()>) {
    use axum::{routing::get, Json, Router};

    let body = serde_json::json!({
        "object": "list",
        "data": models
            .into_iter()
            .map(|id| serde_json::json!({ "id": id, "object": "model" }))
            .collect::<Vec<_>>(),
    });

    let app = Router::new().route("/v1/models", get(move || {
        let body = body.clone();
        async move { Json(body) }
    }));

    // Port 0: the OS picks a free one, so parallel tests cannot collide.
    let listener = tokio::net::TcpListener::bind::<SocketAddr>("127.0.0.1:0".parse().unwrap())
        .await
        .expect("loopback port");
    let port = listener.local_addr().expect("address").port();
    let handle = tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });

    (format!("http://127.0.0.1:{port}/v1"), handle)
}

/// Roughly a 12 GB card — enough for the 7B, and the figure the router plans against.
const VRAM: u64 = 12 * 1024 * 1024 * 1024;

#[tokio::test]
async fn a_coding_task_and_a_document_task_reach_different_models_on_different_runtimes() {
    let dir = tempfile::tempdir().expect("temp dir");
    let (vllm_url, server) = fake_vllm(vec!["granite-docling-258m"]).await;
    let registry = registry_with_both_runtimes(dir.path(), &vllm_url);

    // A coding request. Classified from the words, with no hint from the caller.
    let coding = ModelRouter::route(
        &registry,
        "Write and test a Python function that computes pump efficiency from head and flow.",
        None,
        VRAM,
        None,
        false,
        &[],
        &[],
    )
    .expect("a coding model is available");

    // A document request, routed by known kind rather than by classifying the
    // user's words — OCR on a scanned page is not a question about the prompt.
    let document = ModelRouter::route_for_role(
        &registry,
        ModelRole::DocumentOcr,
        None,
        VRAM,
        None,
        false,
        &[],
        &[],
    )
    .expect("a document model is available");

    assert_eq!(coding.model_id, "qwen2.5-coder-7b");
    assert_eq!(coding.role, ModelRole::Coding);
    assert_eq!(document.model_id, "granite-docling-258m");
    assert_eq!(document.role, ModelRole::DocumentOcr);
    assert_ne!(
        coding.model_id, document.model_id,
        "the two task types must not collapse onto one model"
    );

    // And the decision explains itself, which is what step 10 of the problem
    // statement asks for.
    assert!(
        coding.reasons.iter().any(|r| r.to_lowercase().contains("coding")),
        "coding reasons did not mention the task type: {:?}",
        coding.reasons
    );

    server.abort();
}

#[tokio::test]
async fn each_routed_model_resolves_to_a_loopback_endpoint_on_its_own_runtime() {
    let dir = tempfile::tempdir().expect("temp dir");
    let (vllm_url, server) = fake_vllm(vec!["granite-docling-258m"]).await;
    let registry = registry_with_both_runtimes(dir.path(), &vllm_url);
    let servers = ModelServers::new();

    // --- the Python half, fully exercised -------------------------------
    let document = registry.find("granite-docling-258m").expect("registered");
    let plan = plan_gpu_offload(VRAM, document.weights_bytes, document.context_length, None);
    let endpoint = servers
        .endpoint_for(document, dir.path(), &plan)
        .await
        .expect("the external vLLM endpoint resolves");

    assert_eq!(endpoint.base_url, vllm_url);
    assert_eq!(endpoint.runtime, Runtime::PythonSidecar);
    assert!(
        !endpoint.managed,
        "ARJUN must not claim to manage a vLLM it did not start"
    );

    // --- the C++ half, up to the launch ---------------------------------
    let coding = registry.find("qwen2.5-coder-7b").expect("registered");
    let coding_plan = plan_gpu_offload(VRAM, coding.weights_bytes, coding.context_length, None);
    let launch = plan_launch(
        coding,
        &PathBuf::from("/models/qwen2.5-coder-7b-q4.gguf"),
        // No multimodal projector: this is a text coding model. Arguments have
        // been added to `plan_launch` and not here before, which is how this
        // whole integration test stopped compiling and `npm run
        // test:integration` stopped being green on a clean checkout.
        None,
        &coding_plan,
        18080,
        // `auto_fit` false: in production this is
        // `llama_server_fits_layers_itself()`, which shells out to
        // `llama-server --help` to find out whether this build accepts
        // `--n-gpu-layers auto`. A test that called it would produce a
        // different command line on a machine that has the binary than on one
        // that does not, and there is no llama-server in CI. Passing the
        // answer explicitly keeps the plan deterministic — the same choice
        // every unit test in `serving` makes.
        false,
    );

    // The two arguments that matter for sovereignty and for correctness.
    let arg = |name: &str| {
        launch
            .args
            .windows(2)
            .find(|pair| pair[0] == name)
            .map(|pair| pair[1].clone())
    };
    assert_eq!(arg("--host").as_deref(), Some("127.0.0.1"));
    assert_eq!(arg("--ctx-size").as_deref(), Some("32768"));
    assert_eq!(launch.base_url, "http://127.0.0.1:18080/v1");

    // Both endpoints are on this machine, which is the sovereignty claim
    // restated at the point it would actually be broken.
    for url in [&endpoint.base_url, &launch.base_url] {
        assert_eq!(
            reqwest::Url::parse(url).expect("a URL").host_str(),
            Some("127.0.0.1"),
            "{url} is not loopback"
        );
    }

    server.abort();
}

#[tokio::test]
async fn a_python_model_whose_server_is_down_is_reported_before_a_run_starts() {
    let dir = tempfile::tempdir().expect("temp dir");
    // Port 1 on loopback: reserved, nothing legitimate binds it.
    let registry = registry_with_both_runtimes(dir.path(), "http://127.0.0.1:1/v1");
    let servers = ModelServers::new();

    let document = registry.find("granite-docling-258m").expect("registered");
    let plan = plan_gpu_offload(VRAM, document.weights_bytes, document.context_length, None);
    let error = servers
        .endpoint_for(document, dir.path(), &plan)
        .await
        .expect_err("a dead endpoint must not resolve");

    let message = error.to_string();
    // Naming the endpoint is the point: "model unavailable" sends an operator
    // looking in the wrong place.
    assert!(message.contains("127.0.0.1:1"), "{message}");
    assert!(message.contains("Nothing is listening"), "{message}");
}

#[tokio::test]
async fn a_manifest_may_point_a_gguf_model_at_a_server_someone_else_runs() {
    // OpenClaw supports managed *and* external llama-server, and so does this:
    // a site that already operates a shared llama-server should not have to let
    // ARJUN start a second one.
    let dir = tempfile::tempdir().expect("temp dir");
    let (url, server) = fake_vllm(vec!["qwen2.5-coder-7b"]).await;

    let manifest = serde_json::json!({
        "models": [{
            "id": "qwen2.5-coder-7b",
            "name": "Qwen2.5 Coder 7B",
            "version": "1.0",
            "license": "Apache-2.0",
            "runtime": "llamaCpp",
            "roles": ["coding"],
            "parametersB": 7.0,
            "contextLength": 32768,
            "weightsBytes": 4_700_000_000u64,
            "permittedClassifications": ["internal"],
            "path": "qwen2.5-coder-7b-q4.gguf",
            "serving": { "mode": "external", "baseUrl": url },
            "enabled": true
        }]
    });
    std::fs::write(
        dir.path().join("registry.json"),
        serde_json::to_string(&manifest).unwrap(),
    )
    .unwrap();

    let registry = ModelRegistry::load(dir.path()).expect("loads");
    let entry = registry.find("qwen2.5-coder-7b").expect("registered");
    assert_eq!(
        entry.serving,
        Some(ServingSpec::External { base_url: url.clone() })
    );

    let plan = plan_gpu_offload(VRAM, entry.weights_bytes, entry.context_length, None);
    let endpoint = ModelServers::new()
        .endpoint_for(entry, dir.path(), &plan)
        .await
        .expect("an external llama-server resolves");

    assert_eq!(endpoint.base_url, url);
    assert!(!endpoint.managed, "an external server is not ARJUN's to manage");

    server.abort();
}

#[tokio::test]
async fn a_python_model_with_no_endpoint_is_refused_with_the_line_to_add() {
    let dir = tempfile::tempdir().expect("temp dir");
    let manifest = serde_json::json!({
        "models": [{
            "id": "granite-docling-258m",
            "name": "Granite Docling 258M",
            "version": "1.0",
            "license": "Apache-2.0",
            "runtime": "pythonSidecar",
            "roles": ["documentOcr"],
            "parametersB": 0.258,
            "contextLength": 8192,
            "weightsBytes": 520_000_000u64,
            "permittedClassifications": ["internal"],
            "path": "granite-docling-258m",
            "enabled": true
        }]
    });
    std::fs::write(
        dir.path().join("registry.json"),
        serde_json::to_string(&manifest).unwrap(),
    )
    .unwrap();

    let registry = ModelRegistry::load(dir.path()).expect("loads");
    let entry = registry.find("granite-docling-258m").expect("registered");
    let plan = plan_gpu_offload(VRAM, entry.weights_bytes, entry.context_length, None);

    let error = ModelServers::new()
        .endpoint_for(entry, dir.path(), &plan)
        .await
        .expect_err("ARJUN cannot start a vLLM");

    let message = error.to_string();
    assert!(message.contains("does not start"), "{message}");
    // The message has to contain the fix, not just the complaint.
    assert!(message.contains("\"mode\": \"external\""), "{message}");
}
