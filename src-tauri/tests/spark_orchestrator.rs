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
use sarathi_lib::ai_engine::vram_planner::{plan_gpu_offload_at, ContextChoice, KvPrecision};
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
        revision: None,
        min_llama_build: None,
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

    // Charged at the precision *this* binary will allocate at, which is what
    // `serving` does. Asking for the q8_0 figure on a build that does not
    // accept `-ctk` is the over-commit `KvPrecision` was introduced to remove,
    // and a test that asked for it unconditionally would be asserting the bug.
    let precision = sarathi_lib::serving::llama_server_kv_precision();
    let plan = plan_gpu_offload_at(
        RTX_5060_LAPTOP,
        entry.weights_bytes,
        ContextChoice::Planned(entry.context_length),
        Some(meta.block_count),
        Some(meta.kv_cost()),
        precision,
    );

    assert!(
        plan.full_offload,
        "a 4.4 GB model belongs wholly on an 8 GB card: {}",
        plan.reason
    );
    assert!(
        plan.context_length < MEASURED_OOM_AT,
        "the planner chose {} tokens, and {MEASURED_OOM_AT} was measured to fail allocation          on this card",
        plan.context_length
    );

    // The hybrid reading is worth a long window — but only as long as the cache
    // is actually quantised. Stating the bound per precision keeps this an
    // assertion about the geometry rather than about a flag nobody checked.
    let floor = match precision {
        KvPrecision::Q8_0 => 65_536,
        KvPrecision::Fp16 => 32_768,
    };
    assert!(
        plan.context_length >= floor,
        "at {} the hybrid reading should reach {floor}; {} suggests it was not applied",
        precision.label(),
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
    // The same precision the serving path will launch with, so the command line
    // built below is charged against the memory it will actually take.
    let plan = plan_gpu_offload_at(
        RTX_5060_LAPTOP,
        entry.weights_bytes,
        ContextChoice::Planned(entry.context_length),
        Some(meta.block_count),
        Some(meta.kv_cost()),
        sarathi_lib::serving::llama_server_kv_precision(),
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

// ─────────────────────────────────────────────────────────────────────────────
// The pinned manifest, checked against the bytes on this machine.
// ─────────────────────────────────────────────────────────────────────────────

/// What `config/orchestrator-spark-registry.json` declares, read from the file.
///
/// Read rather than restated, because the point of these assertions is that the
/// shipped declaration matches reality. A copy of the numbers here would test
/// the copy.
fn declared() -> serde_json::Value {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("config")
        .join("orchestrator-spark-registry.json");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("{} must be readable: {error}", path.display()));
    let parsed: serde_json::Value =
        serde_json::from_str(&text).expect("the registry fragment must be valid JSON");
    parsed["models"][0].clone()
}

/// The manifest names an immutable revision and a licence, not an alias.
///
/// ## Why a revision at all
///
/// "The Q8_0 from XHToken/Spark-X2.5-4B-GGUF" names a different file whenever
/// the publisher re-uploads. Evidence that cites an alias cites nothing. The
/// sha256 proves *these* bytes; the revision says where they came from, and the
/// two together are what makes the claim checkable by somebody else later.
#[test]
fn the_manifest_pins_a_revision_a_hash_and_a_licence() {
    let entry = declared();

    assert_eq!(
        entry["revision"].as_str(),
        Some("902d865994943ab9235670e24f01846ee06091f2"),
        "the entry must pin an immutable revision rather than a branch or a download alias"
    );
    assert_eq!(
        entry["sha256"].as_str(),
        Some("5c2c3c190e4337e1016b8593ca8e26e8b18c972200b107385d4ec61a25d9dea2")
    );
    assert_eq!(
        entry["license"].as_str(),
        Some("apache-2.0"),
        "a licence a deployment cannot accept is a blocker, not a footnote"
    );
    assert_eq!(entry["weightsBytes"].as_u64(), Some(4_375_021_152));
    assert_eq!(
        entry["minLlamaBuild"].as_u64(),
        Some(10_828),
        "spark2_5 arrived in llama.cpp b10828, and the entry has to say so for the \
         pre-launch check to have anything to compare against"
    );
    // Nothing is cleared for use until somebody reviews it. Asserted because an
    // entry that shipped with classifications filled in would be a model
    // approved by whoever wrote the file rather than by an administrator.
    assert_eq!(
        entry["permittedClassifications"].as_array().map(Vec::len),
        Some(0),
        "an unreviewed model is cleared for nothing"
    );
}

/// The declared size and hash are the size and hash of the file on this machine.
///
/// Skipped, loudly, when the weights are absent — this is a native gate, and a
/// skip that read as a pass is exactly what this repository's evidence rule
/// forbids.
#[test]
fn the_declared_hash_is_the_hash_of_the_weights_on_this_machine() {
    let Some(path) = weights() else {
        println!(
            "BLOCKED: Spark weights are not on this machine, so the pinned hash is unverified here"
        );
        return;
    };
    let entry = declared();

    let actual_bytes = std::fs::metadata(&path)
        .expect("the weights are readable")
        .len();
    assert_eq!(
        actual_bytes,
        entry["weightsBytes"].as_u64().expect("declared"),
        "the file on disk is not the size the manifest pins"
    );

    let data = std::fs::read(&path).expect("the weights are readable");
    let digest = {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(&data);
        format!("{:x}", hasher.finalize())
    };
    assert_eq!(
        digest,
        entry["sha256"].as_str().expect("declared"),
        "the file on disk is not the file the manifest pins"
    );
    println!("VERIFIED: {actual_bytes} bytes, sha256 {digest}");
}

/// This deployment's `llama-server` is new enough for `spark2_5`.
///
/// The check `ModelServers` makes before launching, made here against the real
/// binary so the report says which build this machine actually has.
#[test]
fn this_machines_llama_server_meets_the_architectures_floor() {
    let Some(build) = sarathi_lib::serving::llama_server_build() else {
        println!(
            "BLOCKED: llama-server could not be asked which build it is, so spark2_5 support \
             is unverified here"
        );
        return;
    };
    println!("llama-server build {build}");

    let mut entry = spark_entry(std::path::Path::new("unused"));
    entry.min_llama_build = Some(10_828);

    match sarathi_lib::serving::check_runtime_supports(&entry) {
        Ok(()) => {
            assert!(build >= 10_828);
            println!("VERIFIED: build {build} >= b10828, so spark2_5 is supported");
        }
        Err(unsupported) => {
            // Not a test failure: an old binary is a real deployment state, and
            // the thing under test is that it is *reported* rather than
            // silently worked around.
            println!("BLOCKED: {}", unsupported.explain());
            assert!(build < 10_828);
        }
    }
}

/// A build older than the floor is refused, and nothing is substituted.
///
/// The negative case, which the binary on this machine cannot produce and so is
/// asserted against the check directly.
#[test]
fn an_old_runtime_is_refused_rather_than_quietly_downgraded() {
    let mut entry = spark_entry(std::path::Path::new("unused"));
    entry.min_llama_build = Some(u32::MAX);

    let refusal = sarathi_lib::serving::check_runtime_supports(&entry)
        .expect_err("a floor no build can meet must refuse");
    let said = refusal.explain();
    assert!(said.contains(&entry.id), "{said}");
    assert!(
        said.contains("not substituted"),
        "an operator has to be told a different model or quantisation was not used: {said}"
    );
}

/// The window the planner picks is charged at the precision the server will
/// actually allocate at.
///
/// ## Why this is a test and not arithmetic
///
/// Because the two used to be decided in different places. `serving` probes
/// `llama-server --help` for `-ctk` and launches without cache quantisation
/// when it is absent; the planner charged `q8_0` regardless. On such a binary
/// the plan sized a window against half the memory the server then took, and
/// the model loaded before dying on its own KV cache.
#[test]
fn the_plan_is_charged_at_the_precision_this_binary_will_actually_use() {
    let Some(path) = weights() else {
        println!("BLOCKED: Spark weights are not on this machine");
        return;
    };
    let entry = spark_entry(&path);
    let meta = gguf_meta::read_gguf_metadata(&path).expect("the header must be readable");
    let precision = sarathi_lib::serving::llama_server_kv_precision();
    println!("this binary's KV cache precision: {}", precision.label());

    let plan = sarathi_lib::ai_engine::vram_planner::plan_gpu_offload_at(
        RTX_5060_LAPTOP,
        entry.weights_bytes,
        ContextChoice::Planned(entry.context_length),
        Some(meta.block_count),
        Some(meta.kv_cost()),
        precision,
    );
    println!(
        "planned: {} tokens, full_offload={}, {}",
        plan.context_length, plan.full_offload, plan.reason
    );

    // Whatever the precision, the window has to be one this card can pay for.
    const MEASURED_OOM_AT: u32 = 196_608;
    assert!(
        plan.context_length < MEASURED_OOM_AT,
        "{MEASURED_OOM_AT} was measured to fail allocation on this card"
    );

    // And an f16 cache must never be planned as though it were quantised.
    let pessimistic = sarathi_lib::ai_engine::vram_planner::plan_gpu_offload_at(
        RTX_5060_LAPTOP,
        entry.weights_bytes,
        ContextChoice::Planned(entry.context_length),
        Some(meta.block_count),
        Some(meta.kv_cost()),
        sarathi_lib::ai_engine::vram_planner::KvPrecision::Fp16,
    );
    assert!(
        pessimistic.context_length <= plan.context_length,
        "the full-precision plan must never be the larger of the two"
    );
}

/// The trained maximum, the configured window and the served window are three
/// different numbers and must not be collapsed into one.
#[test]
fn trained_configured_and_served_windows_stay_distinct() {
    let Some(path) = weights() else {
        println!("BLOCKED: Spark weights are not on this machine");
        return;
    };
    let entry = spark_entry(&path);
    let meta = gguf_meta::read_gguf_metadata(&path).expect("the header must be readable");

    // Trained: what the file says it was trained for.
    assert_eq!(meta.context_length, Some(1_048_576));
    // Configured: what the registry entry declares, which is the ceiling the
    // planner may walk *down* from — never a window to allocate outright.
    assert_eq!(entry.context_length, 1_048_576);

    // Served: what this card can pay for, which is neither of the above.
    let served = sarathi_lib::ai_engine::vram_planner::plan_gpu_offload_at(
        RTX_5060_LAPTOP,
        entry.weights_bytes,
        ContextChoice::Planned(entry.context_length),
        Some(meta.block_count),
        Some(meta.kv_cost()),
        sarathi_lib::serving::llama_server_kv_precision(),
    )
    .context_length;

    assert!(
        served < entry.context_length,
        "a 7.9 GB card cannot serve the trained window, and a plan that said it could \
         would be allocating 1M tokens because a card advertised it"
    );
    assert!(
        served >= 16_384,
        "the served window must still be a usable one: got {served}"
    );
    println!(
        "trained 1048576 / configured {} / served {served}",
        entry.context_length
    );
}

/// What admission charged for and what the server is told to allocate are the
/// same decision, read from the same probe.
///
/// ## The failure this closes
///
/// `serving::plan_launch` adds `-ctk q8_0 -ctv q8_0` only when the binary's
/// `--help` mentions `-ctk`. The planner applied a flat 0.5 KV factor whatever
/// the binary was. On a build without those flags the plan therefore sized the
/// window against half the memory the server went on to take — and the failure
/// arrives as `failed to allocate buffer for kv cache`, after several seconds
/// and several gigabytes, reading like a model problem.
///
/// Asserting the two agree is the only way to keep them agreeing: they are
/// computed in different modules and nothing else would notice them drifting.
#[test]
fn what_admission_charges_for_is_what_the_launch_asks_for() {
    let Some(path) = weights() else {
        println!("BLOCKED: Spark weights are not on this machine");
        return;
    };
    if Command::new("llama-server")
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_err()
    {
        println!("BLOCKED: llama-server is not on this machine");
        return;
    }

    let entry = spark_entry(&path);
    let meta = gguf_meta::read_gguf_metadata(&path).expect("the header must be readable");
    let precision = sarathi_lib::serving::llama_server_kv_precision();
    let plan = plan_gpu_offload_at(
        RTX_5060_LAPTOP,
        entry.weights_bytes,
        ContextChoice::Planned(entry.context_length),
        Some(meta.block_count),
        Some(meta.kv_cost()),
        precision,
    );
    let launch = plan_launch(&entry, &path, None, &plan, 8080, true);

    let quantises = launch
        .args
        .windows(2)
        .any(|pair| pair[0] == "-ctk" && pair[1] == "q8_0");
    match precision {
        KvPrecision::Q8_0 => assert!(
            quantises,
            "admission charged a quantised cache and the launch does not ask for one: {:?}",
            launch.args
        ),
        KvPrecision::Fp16 => assert!(
            !quantises,
            "admission charged a full-precision cache and the launch asks for a quantised              one, so the server will take less than was reserved: {:?}",
            launch.args
        ),
    }

    // And the window the plan chose is the window the server is started with.
    // Two halves of one decision; a launch that took its own figure would make
    // the admission arithmetic describe a server that does not exist.
    let served: u32 = launch
        .args
        .windows(2)
        .find(|pair| pair[0] == "--ctx-size" || pair[0] == "-c")
        .and_then(|pair| pair[1].parse().ok())
        .expect("the launch must state a context size");
    assert_eq!(
        served, plan.context_length,
        "the planner sized {} and the server is told {served}",
        plan.context_length
    );
    println!(
        "admission {} @ {} tokens == launch -ctk {} @ {served} tokens",
        precision.label(),
        plan.context_length,
        if quantises { "q8_0" } else { "absent" }
    );
}
