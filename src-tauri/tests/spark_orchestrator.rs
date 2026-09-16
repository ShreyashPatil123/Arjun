//! End-to-end: ARJUN's own decisions about Spark-X2.5-4B, checked against the
//! server they are decisions about.
//!
//! Every other test of this work is arithmetic — the KV cost model, the context
//! ladder, the launch-plan arguments. Arithmetic that is internally consistent
//! and wrong about the machine is exactly the failure this repository has a
//! rule against, so this test closes the loop: it reads the real header, plans
//! against the real card, builds the real command line, **starts it**, and asks
//! the model a question.
//!
//! ## Why this is skipped rather than failed when the model is absent
//!
//! The weights are 4.4 GB and are not in the repository. A machine without them
//! cannot run this and is not broken, so it says so and returns — the judgement
//! `scan_real_library` makes about a machine with no model library. A machine
//! that *has* the model gets no such leniency: every assertion below then runs
//! for real.
//!
//! Set `ARJUN_SPARK_GGUF` to run it against weights kept somewhere else.

use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use sarathi_lib::ai_engine::gguf_meta;
use sarathi_lib::ai_engine::vram_planner::{plan_gpu_offload_with, ContextChoice};
use sarathi_lib::registry::{
    Modality, ModelEntry, ModelRole, RoutingPreference, Runtime, SamplingDefaults,
};
use sarathi_lib::serving::plan_launch;

const DEFAULT_PATH: &str = r"C:\Users\lenovo\models\Spark-X2.5-4B-Q8\Spark-X2.5-4B-Q8_0.gguf";

/// The card this was developed against, stated rather than measured.
///
/// The assertions are about the planner's arithmetic; reading live VRAM would
/// make them pass or fail on whatever else happened to be running.
const RTX_5060_LAPTOP: u64 = 7899 * 1024 * 1024;

fn weights() -> Option<PathBuf> {
    let path = std::env::var("ARJUN_SPARK_GGUF")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(DEFAULT_PATH));
    path.is_file().then_some(path)
}

/// The registry entry `config/orchestrator-spark-registry.json` declares.
///
/// Built here rather than read from that file, so a disagreement between the
/// two about anything load-bearing fails rather than testing the file against
/// itself.
fn spark_entry(path: &std::path::Path) -> ModelEntry {
    ModelEntry {
        id: "orchestrator.spark-x2-5-4b".into(),
        name: "Spark-X2.5 4B (Q8_0)".into(),
        version: "2.5".into(),
        license: "apache-2.0".into(),
        sha256: None,
        runtime: Runtime::LlamaCpp,
        roles: vec![ModelRole::Reasoning, ModelRole::Coding],
        modalities: vec![Modality::Text],
        quantization: Some("Q8_0".into()),
        parameters_b: 4.1,
        active_parameters_b: None,
        context_length: 1_048_576,
        weights_bytes: std::fs::metadata(path).map(|m| m.len()).unwrap_or(0),
        supports_structured_output: true,
        permitted_classifications: vec![],
        path: path.to_path_buf(),
        projector: None,
        load: None,
        serving: None,
        required_runtime_profile: None,
        enabled: true,
        routing: RoutingPreference::default(),
        sampling: Some(SamplingDefaults {
            temperature: Some(0.15),
            top_p: Some(0.95),
            top_k: None,
        }),
    }
}

/// The header says what the planner was told it says.
#[test]
fn the_real_header_declares_the_hybrid_attention_the_planner_relies_on() {
    let Some(path) = weights() else {
        println!("Spark weights are not on this machine; nothing to check");
        return;
    };

    let meta = gguf_meta::read_gguf_metadata(&path).expect("the header must be readable");
    assert_eq!(meta.architecture, "spark2_5");
    assert_eq!(meta.block_count, 36);
    assert_eq!(
        meta.full_attention_layers,
        Some(9),
        "9 of 36 blocks attend to the whole context; every window figure follows from this"
    );
    assert_eq!(meta.sliding_window, Some(512));
    assert_eq!(
        meta.context_length,
        Some(1_048_576),
        "the trained window the registry entry declares"
    );
    assert!(
        meta.emits_reasoning,
        "Spark reasons, and a runtime told otherwise holds its answer back instead of streaming"
    );

    let cost = meta.kv_cost();
    assert_eq!(cost.per_token, 36_864);
    assert!(
        cost.per_token * 4 == meta.kv_bytes_per_token(),
        "reading this model as dense over-charges it four-fold, which is what the hybrid \
         reading removes"
    );
}

/// The planner's answer on this machine is one this machine can actually pay.
///
/// Measured directly before this work: `llama-server` loads Spark at 163 840
/// tokens on the 7.9 GB card and fails to allocate its KV cache at 196 608. A
/// plan above that ceiling is a plan that dies in `ggml_vulkan`, so whatever
/// the planner picks has to sit under it.
#[test]
fn the_plan_this_machine_gets_is_one_it_can_pay_for() {
    let Some(path) = weights() else {
        println!("Spark weights are not on this machine; nothing to check");
        return;
    };
    let entry = spark_entry(&path);
    let meta = gguf_meta::read_gguf_metadata(&path).expect("the header must be readable");
    const MEASURED_OOM_AT: u32 = 196_608;

    let plan = plan_gpu_offload_with(
        RTX_5060_LAPTOP,
        entry.weights_bytes,
        ContextChoice::Planned(entry.context_length),
        Some(meta.block_count),
        Some(meta.kv_cost()),
    );

    assert!(
        plan.full_offload,
        "a 4.4 GB model belongs wholly on an 8 GB card: {}",
        plan.reason
    );
    assert!(
        plan.context_length < MEASURED_OOM_AT,
        "the planner chose {} tokens, and {MEASURED_OOM_AT} was measured to fail allocation \
         on this card",
        plan.context_length
    );
    assert!(
        plan.context_length >= 65_536,
        "the hybrid reading is worth a long window; {} suggests it was not applied",
        plan.context_length
    );
}

/// The command line ARJUN builds starts a server that answers.
///
/// The one assertion arithmetic cannot make. It is also the one that would have
/// caught the specification this work started from, whose literal command line
/// fails on this machine with `failed to allocate buffer for kv cache`.
#[test]
fn the_command_line_arjun_builds_starts_a_server_that_answers() {
    let Some(path) = weights() else {
        println!("Spark weights are not on this machine; nothing to check");
        return;
    };
    if Command::new("llama-server")
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_err()
    {
        println!("llama-server is not on this machine; nothing to start");
        return;
    }

    let entry = spark_entry(&path);
    let meta = gguf_meta::read_gguf_metadata(&path).expect("the header must be readable");
    let plan = plan_gpu_offload_with(
        RTX_5060_LAPTOP,
        entry.weights_bytes,
        ContextChoice::Planned(entry.context_length),
        Some(meta.block_count),
        Some(meta.kv_cost()),
    );

    // A port of our own. The default 8080 is where an operator's own server
    // usually is, and a test that fought it would be reporting their server's
    // health rather than this plan's.
    let port = std::net::TcpListener::bind("127.0.0.1:0")
        .expect("a loopback port")
        .local_addr()
        .expect("a bound address")
        .port();

    let launch = plan_launch(&entry, &path, None, &plan, port, true);

    // The sampling the entry declares reaches the command line.
    assert!(
        launch
            .args
            .windows(2)
            .any(|pair| pair[0] == "--temp" && pair[1] == "0.15"),
        "the orchestrator's temperature must be served: {:?}",
        launch.args
    );

    // And so does the device, when one was named. Asserted rather than assumed
    // from the server starting: it would start without the flag too, so a
    // passing launch proves nothing about whether the card was pinned.
    if let Ok(device) = std::env::var("ARJUN_LLAMA_DEVICE") {
        let device = device.trim().to_string();
        if !device.is_empty() {
            assert!(
                launch
                    .args
                    .windows(2)
                    .any(|pair| pair[0] == "--device" && pair[1] == device),
                "ARJUN_LLAMA_DEVICE={device} was set, so the launch must pin that device: {:?}",
                launch.args
            );
        }
    }

    let mut child = Command::new(&launch.program)
        .args(&launch.args)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("llama-server must start");

    // Generous: a cold 4.4 GB model on a laptop GPU is not instant, and a tight
    // timeout here would be a test that fails for being slow.
    let deadline = Instant::now() + Duration::from_secs(240);
    let url = format!("http://127.0.0.1:{port}/v1/chat/completions");
    let mut answered = None;
    while Instant::now() < deadline {
        if let Ok(Some(status)) = child.try_wait() {
            let _ = child.wait();
            panic!("llama-server exited before answering, with status {status}");
        }
        if let Some(body) = post_chat(&url) {
            answered = Some(body);
            break;
        }
        std::thread::sleep(Duration::from_secs(2));
    }

    let _ = child.kill();
    let _ = child.wait();

    let body = answered.expect("the planned server must become ready and answer within 240s");
    assert!(
        body.contains("choices"),
        "the server answered something that is not a chat completion: {body}"
    );
}

/// A minimal chat request, returning the body when the server answered.
///
/// Written by hand rather than with an HTTP client: this crate's test
/// dependencies are not the place to add one for a single POST, and the request
/// is three lines of HTTP/1.1.
fn post_chat(url: &str) -> Option<String> {
    use std::io::{Read, Write};

    let rest = url.strip_prefix("http://")?;
    let address = rest.split('/').next()?.to_string();
    let path = rest.strip_prefix(&address)?;
    let body = r#"{"messages":[{"role":"user","content":"Reply with the single word READY."}],"max_tokens":32,"stream":false}"#;
    let request = format!(
        "POST {path} HTTP/1.1\r\nHost: {address}\r\nContent-Type: application/json\r\n\
         Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );

    let mut stream = std::net::TcpStream::connect(&address).ok()?;
    stream
        .set_read_timeout(Some(Duration::from_secs(120)))
        .ok()?;
    stream.write_all(request.as_bytes()).ok()?;
    let mut response = String::new();
    stream.read_to_string(&mut response).ok()?;
    response.contains(" 200 ").then_some(response)
}
