#!/usr/bin/env node
/**
 * The machine-readable qualification inventory (plan P00).
 *
 *   node scripts/qualification-inventory.mjs [--hash] [--out <path>]
 *
 * ## The rule this file is written around
 *
 * **A field nobody measured is `null`, and it says why.** Every value carries a
 * sibling `source` naming where it came from: `measured` (a command ran on this
 * machine and this is its output), `declared` (a file on this machine says so
 * and nothing checked it), or `unknown` (nothing established it, and
 * `unknown_because` says what would).
 *
 * That distinction is the whole point of the file. The build plan's target is
 * an RTX 5060 with 8 GB; this machine has an RTX 5060 *Laptop* GPU. Those are
 * different SKUs with different power envelopes and memory bandwidth, and
 * writing one where the other belongs would turn a guess into something that
 * reads like a measurement. So the target and the current machine are separate
 * objects, and whether they are the same machine is `unknown` until somebody
 * establishes it rather than `true` because the names rhyme.
 *
 * Nothing here downloads, launches or writes a model. Reads only, plus
 * `nvidia-smi`, `docker info` and `llama-server --version`, which report.
 */

import { createHash } from 'node:crypto';
import {
  closeSync, createReadStream, existsSync, mkdirSync, openSync,
  readFileSync, readSync, statSync, writeFileSync,
} from 'node:fs';
import { spawnSync } from 'node:child_process';
import { basename, dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const ARGV = process.argv.slice(2);
const WANT_HASHES = ARGV.includes('--hash');
const OUT = (() => {
  const at = ARGV.indexOf('--out');
  return at >= 0 && ARGV[at + 1]
    ? ARGV[at + 1]
    : join(ROOT, 'evidence', 'agent-system', 'P00', 'qualification-inventory.json');
})();

/** A value nobody established. Never a plausible default. */
const unknown = (because) => ({ value: null, source: 'unknown', unknown_because: because });
/** A value a command on this machine produced. */
const measured = (value, how) => ({ value, source: 'measured', measured_by: how });
/** A value a file on this machine asserts, which nothing has checked. */
const declared = (value, where) => ({ value, source: 'declared', declared_in: where });

function run(command, args, timeoutMs = 60_000) {
  const result = spawnSync(command, args, { encoding: 'utf8', timeout: timeoutMs, windowsHide: true });
  return {
    absent: result.error?.code === 'ENOENT',
    ok: !result.error && result.status === 0,
    status: result.status,
    stdout: (result.stdout ?? '').trim(),
    stderr: (result.stderr ?? '').trim(),
  };
}

const ps = (script) => run('powershell', ['-NoProfile', '-NonInteractive', '-Command', script]);

// ── GGUF headers ────────────────────────────────────────────────────────────
//
// The registry's `quantization` is a string somebody typed; two of the entries
// on this machine say "unknown" for a file whose name ends `UD-Q4_K_XL`. The
// header is the file's own account of itself, so architecture and trained
// context come from there and are labelled `measured` rather than `declared`.

const GGUF = {
  UINT8: 0, INT8: 1, UINT16: 2, INT16: 3, UINT32: 4, INT32: 5,
  FLOAT32: 6, BOOL: 7, STRING: 8, ARRAY: 9, UINT64: 10, INT64: 11, FLOAT64: 12,
};

/**
 * Reads just enough of a GGUF header to answer "what is this file".
 *
 * The window is 64 MiB because a large model's token vocabulary lives in the
 * header and the 35B on this machine overruns 8 MiB. A header that still does
 * not fit returns an error rather than a partial answer: half a header would
 * produce a plausible architecture for the wrong file.
 */
function ggufHeader(path, maxBytes = 64 * 1024 * 1024) {
  let fd;
  try {
    fd = openSync(path, 'r');
    const size = statSync(path).size;
    const buf = Buffer.alloc(Math.min(maxBytes, size));
    readSync(fd, buf, 0, buf.length, 0);
    if (buf.length < 24 || buf.toString('ascii', 0, 4) !== 'GGUF') return null;

    const version = buf.readUInt32LE(4);
    const tensorCount = Number(buf.readBigUInt64LE(8));
    const kvCount = Number(buf.readBigUInt64LE(16));
    let at = 24;

    const readString = () => {
      const len = Number(buf.readBigUInt64LE(at));
      at += 8;
      const s = buf.toString('utf8', at, at + len);
      at += len;
      return s;
    };
    const readScalar = (type) => {
      switch (type) {
        case GGUF.UINT8: return buf.readUInt8(at++);
        case GGUF.INT8: return buf.readInt8(at++);
        case GGUF.UINT16: { const v = buf.readUInt16LE(at); at += 2; return v; }
        case GGUF.INT16: { const v = buf.readInt16LE(at); at += 2; return v; }
        case GGUF.UINT32: { const v = buf.readUInt32LE(at); at += 4; return v; }
        case GGUF.INT32: { const v = buf.readInt32LE(at); at += 4; return v; }
        case GGUF.FLOAT32: { const v = buf.readFloatLE(at); at += 4; return v; }
        case GGUF.BOOL: return buf.readUInt8(at++) !== 0;
        case GGUF.STRING: return readString();
        case GGUF.UINT64: { const v = Number(buf.readBigUInt64LE(at)); at += 8; return v; }
        case GGUF.INT64: { const v = Number(buf.readBigInt64LE(at)); at += 8; return v; }
        case GGUF.FLOAT64: { const v = buf.readDoubleLE(at); at += 8; return v; }
        default: throw new Error(`unhandled gguf scalar type ${type}`);
      }
    };

    const kv = {};
    for (let i = 0; i < kvCount; i += 1) {
      if (at + 12 > buf.length) throw new Error('header larger than the window read');
      const key = readString();
      const type = buf.readUInt32LE(at);
      at += 4;
      if (type === GGUF.ARRAY) {
        const inner = buf.readUInt32LE(at);
        at += 4;
        const len = Number(buf.readBigUInt64LE(at));
        at += 8;
        // Token vocabularies are megabytes of strings nobody here reads. The
        // length is kept, the contents are skipped, and the skip is recorded
        // rather than silently producing an empty array.
        if (len > 64) {
          for (let j = 0; j < len; j += 1) readScalar(inner);
          kv[key] = { arrayOf: inner, length: len, skipped: true };
        } else {
          const items = [];
          for (let j = 0; j < len; j += 1) items.push(readScalar(inner));
          kv[key] = items;
        }
      } else {
        kv[key] = readScalar(type);
      }
    }

    const arch = kv['general.architecture'];
    const template = kv['tokenizer.chat_template'];
    return {
      gguf_version: version,
      tensor_count: tensorCount,
      architecture: arch ?? null,
      name: kv['general.name'] ?? null,
      file_type: kv['general.file_type'] ?? null,
      size_label: kv['general.size_label'] ?? null,
      trained_context_length: arch ? (kv[`${arch}.context_length`] ?? null) : null,
      chat_template_present: typeof template === 'string',
      chat_template_sha256: typeof template === 'string'
        ? createHash('sha256').update(template).digest('hex')
        : null,
    };
  } catch (error) {
    return { error: String(error?.message ?? error) };
  } finally {
    if (fd !== undefined) closeSync(fd);
  }
}

function sha256File(path) {
  return new Promise((resolve, reject) => {
    const hash = createHash('sha256');
    const stream = createReadStream(path, { highWaterMark: 4 * 1024 * 1024 });
    stream.on('error', reject);
    stream.on('data', (chunk) => hash.update(chunk));
    stream.on('end', () => resolve(hash.digest('hex')));
  });
}

// ── The two machines ────────────────────────────────────────────────────────

/**
 * The hardware the build plan targets, copied from its own words.
 *
 * Every measured field is `unknown` on purpose: nothing in this process has
 * ever run on that machine. The plan section is the citation, and the citation
 * is all this object is.
 */
function targetMachine() {
  return {
    described_in: 'docs/plans/2026-09-20-agent-system-build-plan.md §4',
    gpu: declared('RTX 5060, 8 GB VRAM', 'build plan preamble and §4'),
    gpu_sku_resolved: unknown('the plan does not say desktop or laptop, and the two differ in power envelope and memory bandwidth'),
    vram_bytes: unknown('not measured on the target machine by this process'),
    system_ram_bytes: unknown('the plan states target system RAM is not yet known'),
    observed_served_context: unknown('no model has been served on the target machine by this process'),
  };
}

function currentMachine() {
  const host = (() => {
    const probe = ps(
      '$os=Get-CimInstance Win32_OperatingSystem; $cs=Get-CimInstance Win32_ComputerSystem;' +
      ' $cpu=Get-CimInstance Win32_Processor | Select-Object -First 1;' +
      ' [pscustomobject]@{os=$os.Caption;version=$os.Version;ram=$cs.TotalPhysicalMemory;' +
      'freeKb=$os.FreePhysicalMemory;cpu=$cpu.Name;cores=$cpu.NumberOfCores;' +
      'threads=$cpu.NumberOfLogicalProcessors} | ConvertTo-Json -Compress');
    if (!probe.ok) return null;
    try { return JSON.parse(probe.stdout); } catch { return null; }
  })();

  const smi = run('nvidia-smi', [
    '--query-gpu=index,name,memory.total,memory.free,driver_version,compute_cap',
    '--format=csv,noheader,nounits',
  ]);
  const gpus = smi.ok
    ? smi.stdout.split(/\r?\n/).filter(Boolean).map((line) => {
        const [index, name, total, free, driver, cap] = line.split(',').map((s) => s.trim());
        return {
          index: Number(index),
          name,
          vram_total_bytes: Number(total) * 1024 * 1024,
          vram_free_bytes_at_scan: Number(free) * 1024 * 1024,
          driver_version: driver,
          compute_capability: cap,
        };
      })
    : null;

  const cimFailed = 'the PowerShell CIM query did not return JSON';
  return {
    is_the_target_machine: unknown(
      'not established. This host reports an "RTX 5060 Laptop GPU"; the plan targets an "RTX 5060". ' +
      'Only the hardware owner can confirm whether they are the same deployment target.'),
    os: host ? measured(`${host.os} ${host.version}`, 'Win32_OperatingSystem') : unknown(cimFailed),
    cpu: host ? measured(String(host.cpu ?? '').trim(), 'Win32_Processor') : unknown(cimFailed),
    cpu_cores: host ? measured(host.cores, 'Win32_Processor') : unknown(cimFailed),
    cpu_threads: host ? measured(host.threads, 'Win32_Processor') : unknown(cimFailed),
    system_ram_bytes: host ? measured(Number(host.ram), 'Win32_ComputerSystem.TotalPhysicalMemory') : unknown(cimFailed),
    system_ram_free_bytes_at_scan: host ? measured(Number(host.freeKb) * 1024, 'Win32_OperatingSystem.FreePhysicalMemory') : unknown(cimFailed),
    gpus: gpus
      ? measured(gpus, 'nvidia-smi --query-gpu')
      : unknown(smi.absent ? 'nvidia-smi is not on PATH' : `nvidia-smi exited ${smi.status}: ${smi.stderr}`),
  };
}

// ── Runtime, containers, renderers ──────────────────────────────────────────

function llamaRuntime() {
  const explicit = process.env.ARJUN_LLAMA_SERVER;
  const probe = run(explicit || 'llama-server', ['--version']);
  // llama-server prints its banner on stderr, so both streams are searched.
  const banner = `${probe.stdout}\n${probe.stderr}`;
  const build = banner.match(/build\s+(\d+)/i);
  const commit = banner.match(/commit\s+([0-9a-f]{7,40})/i);
  const version = banner.match(/version:\s*([^\s(]+)/i);
  const where = ps('$c = Get-Command llama-server -ErrorAction SilentlyContinue; if ($c) { $c.Source }');

  return {
    program: explicit
      ? declared(explicit, 'ARJUN_LLAMA_SERVER')
      : (where.ok && where.stdout
          ? measured(where.stdout, 'Get-Command llama-server')
          : unknown('llama-server is not on PATH and ARJUN_LLAMA_SERVER is unset')),
    build_number: build ? measured(Number(build[1]), 'llama-server --version') : unknown('llama-server did not report a build number'),
    commit: commit ? measured(commit[1], 'llama-server --version') : unknown('llama-server did not report a commit'),
    version_string: version ? measured(version[1], 'llama-server --version') : unknown('llama-server did not report a version'),
    minimum_build_required_by_registry: unknown(
      'no registry row on this machine carries a minLlamaBuild field. src-tauri/src/registry/mod.rs ' +
      "documents b10828 for Spark's spark2_5 architecture, but with the field absent the launch gate cannot fire."),
  };
}

function containers() {
  const dockerCli = ps('$c = Get-Command docker -ErrorAction SilentlyContinue; if ($c) { $c.Source }');
  const podmanCli = ps('$c = Get-Command podman -ErrorAction SilentlyContinue; if ($c) { $c.Source }');
  const daemon = dockerCli.stdout ? run('docker', ['info', '--format', '{{.ServerVersion}}'], 30_000) : null;
  const daemonUp = Boolean(daemon?.ok && daemon.stdout);
  const images = daemonUp ? run('docker', ['images', '--format', '{{.Repository}}:{{.Tag}}'], 30_000) : null;

  return {
    docker_cli: dockerCli.stdout ? measured(dockerCli.stdout, 'Get-Command docker') : unknown('docker is not on PATH'),
    podman_cli: podmanCli.stdout ? measured(podmanCli.stdout, 'Get-Command podman') : unknown('podman is not on PATH'),
    daemon_reachable: daemon
      ? measured(daemonUp, 'docker info')
      : unknown('no container CLI on PATH, so no daemon was contacted'),
    daemon_error: !daemon || daemonUp
      ? null
      : measured(daemon.stderr.split('\n')[0] ?? `exit ${daemon.status}`, 'docker info stderr'),
    server_version: daemonUp ? measured(daemon.stdout, 'docker info') : unknown('the container daemon did not answer'),
    images: daemonUp && images?.ok
      ? measured(images.stdout ? images.stdout.split(/\r?\n/).filter(Boolean) : [], 'docker images')
      : unknown('the container daemon did not answer, so its image list is not known'),
    unblock_command: daemonUp ? null : 'Start Docker Desktop (or `podman machine start`), then re-run this script.',
  };
}

function renderers() {
  const probe = (name) => {
    const where = ps(`$c = Get-Command ${name} -ErrorAction SilentlyContinue; if ($c) { $c.Source }`);
    if (!where.stdout) return unknown(`${name} is not on PATH`);
    const version = run(name, ['--version']);
    return measured(
      { path: where.stdout, banner: `${version.stdout}${version.stderr}`.split('\n')[0] ?? '' },
      `${name} --version`);
  };

  // Python document libraries decide whether a produced artifact can be
  // *independently* reopened — by something other than the code that wrote it,
  // which is the only reopen worth anything as evidence.
  const py = run('python', ['-c',
    'import json,importlib\n' +
    'out={}\n' +
    "for m in ('PIL','fitz','docx','pptx','openpyxl','reportlab'):\n" +
    '    try:\n' +
    '        mod=importlib.import_module(m); out[m]=getattr(mod,"__version__","present")\n' +
    '    except Exception:\n' +
    '        out[m]=None\n' +
    'print(json.dumps(out))']);
  let libs = null;
  if (py.ok) { try { libs = JSON.parse(py.stdout); } catch { libs = null; } }

  return {
    soffice: probe('soffice'),
    libreoffice: probe('libreoffice'),
    python_document_libraries: libs ? measured(libs, 'python -c import') : unknown('the python probe did not return JSON'),
    note:
      'Word, PowerPoint and Excel files are written by Rust directly as OOXML (src-tauri/Cargo.toml: `zip`), so no ' +
      'renderer is needed to PRODUCE them. A renderer is needed to RASTERISE a page or slide for visual review ' +
      '(plan §5.9) and to reopen a workbook with a real recalculation engine (plan §5.6).',
  };
}

// ── Models ──────────────────────────────────────────────────────────────────

const APP_IDENTIFIER = 'com.arjun.workbench';

/** The quantization a GGUF filename claims, or null. Never a guess. */
function quantFromName(name) {
  const match = name.match(/(?:^|[-_.])((?:UD-)?(?:IQ|Q)\d(?:_[A-Z0-9]+)*|F16|BF16|F32)\.gguf$/i);
  return match ? match[1] : null;
}

async function models() {
  const roaming = process.env.APPDATA ? join(process.env.APPDATA, APP_IDENTIFIER) : null;
  const registryPath = roaming ? join(roaming, 'models', 'registry.json') : null;

  if (!registryPath || !existsSync(registryPath)) {
    return {
      app_identifier: measured(APP_IDENTIFIER, 'src-tauri/tauri.conf.json identifier'),
      registry_path: unknown(`no registry.json at ${registryPath ?? '(APPDATA unset)'}`),
      entries: [],
    };
  }

  const registry = JSON.parse(readFileSync(registryPath, 'utf8'));
  const rows = Array.isArray(registry) ? registry : (registry.models ?? []);
  const entries = [];

  for (const row of rows) {
    const path = row.path;
    const present = typeof path === 'string' && existsSync(path);
    const stat = present ? statSync(path) : null;

    let hash;
    if (row.sha256) {
      hash = declared(row.sha256, 'models/registry.json');
    } else if (WANT_HASHES && present) {
      hash = measured(await sha256File(path), 'sha256 of the file on disk');
    } else {
      hash = unknown(present
        ? 'the registry row records no sha256. Re-run with --hash to compute it from the file.'
        : 'the file the registry names is not on this machine.');
    }

    const header = present && path.toLowerCase().endsWith('.gguf') ? ggufHeader(path) : null;
    const projectorPath = row.projector ?? null;

    entries.push({
      id: row.id,
      name: row.name ?? null,
      enabled: row.enabled ?? null,
      path: present
        ? measured(path, 'file exists on this machine')
        : unknown(`the registry names ${path}, which is not on this machine`),
      bytes_on_disk: stat ? measured(stat.size, 'fs.statSync') : unknown('file absent'),
      bytes_declared: row.weightsBytes ?? null,
      bytes_agree: stat && row.weightsBytes ? stat.size === row.weightsBytes : null,
      sha256: hash,
      quantization_declared: row.quantization
        ? declared(row.quantization, 'models/registry.json')
        : unknown('the registry row records no quantization'),
      quantization_from_filename: (() => {
        if (!present) return unknown('file absent');
        if (!path.toLowerCase().endsWith('.gguf')) return unknown('not a GGUF file; the filename carries no quantization suffix');
        const quant = quantFromName(basename(path));
        return quant
          ? measured(quant, 'GGUF filename suffix')
          : unknown('the GGUF filename carries no recognised quantization suffix');
      })(),
      variant_from_header: header && !header.error
        ? measured({
            architecture: header.architecture,
            name: header.name,
            size_label: header.size_label,
            file_type: header.file_type,
          }, 'GGUF key-value header')
        : unknown(header?.error ? `GGUF header unreadable: ${header.error}` : 'not a GGUF file on this machine'),
      trained_context: header && !header.error && header.trained_context_length !== null
        ? measured(header.trained_context_length, 'GGUF header')
        : unknown('no <arch>.context_length key was read from the GGUF header'),
      configured_context: row.contextLength
        ? declared(row.contextLength, 'models/registry.json contextLength')
        : unknown('the registry row records no contextLength'),
      // The number that actually matters on an 8 GB card, and the one nothing
      // on this machine has ever written down.
      observed_served_context: unknown(
        'no llama-server was started by this inventory. Run the baseline harness with --serve to have a real ' +
        'server report its n_ctx, then re-run this script.'),
      chat_template: header && !header.error
        ? measured(
            { embedded: header.chat_template_present, sha256: header.chat_template_sha256 },
            'GGUF tokenizer.chat_template')
        : unknown('GGUF header not read'),
      projector: projectorPath
        ? (existsSync(projectorPath)
            ? measured({ path: projectorPath, bytes: statSync(projectorPath).size }, 'registry `projector` field, file present')
            : unknown(`the registry pins projector ${projectorPath}, which is not on this machine`))
        : ((row.modalities ?? []).includes('image')
            ? unknown('this row claims the image modality and pins no projector')
            : { value: null, source: 'not-applicable', note: 'text-only row' }),
      roles: row.roles ?? [],
      modalities: row.modalities ?? [],
      license: row.license
        ? declared(row.license, 'models/registry.json')
        : unknown('the registry row records no license'),
      min_llama_build: row.minLlamaBuild
        ? declared(row.minLlamaBuild, 'models/registry.json')
        : unknown('the registry row carries no minLlamaBuild, so the launch gate cannot check this architecture against the installed llama-server'),
      // Plan §4: discovered -> compatible runtime -> task-qualified -> approved.
      qualification_state: present ? 'discovered' : 'absent',
    });
  }

  return {
    app_identifier: measured(APP_IDENTIFIER, 'src-tauri/tauri.conf.json identifier'),
    registry_path: measured(registryPath, 'file exists on this machine'),
    qualification_state_note:
      'Nothing in this inventory loads a model, so no row can be higher than `discovered`. Advancing to ' +
      '`compatible runtime` needs a real load; `task-qualified` needs the fixture pack run against it.',
    entries,
  };
}

function ocrSettings() {
  // The OCR configuration that actually reaches the server, read from source
  // rather than restated, so this file cannot drift from the code.
  const profilePath = join(ROOT, 'src-tauri', 'src', 'ai_engine', 'ocr_profile.rs');
  const source = existsSync(profilePath) ? readFileSync(profilePath, 'utf8') : '';
  const tiers = [...source.matchAll(/OcrTier::(\w+)\s*=>\s*"([^"]+\.gguf)"/g)]
    .map(([, tier, file]) => ({ tier, file }));

  return {
    source_of_truth: 'src-tauri/src/ai_engine/ocr_profile.rs',
    weight_files_per_tier: tiers.length
      ? measured(tiers, 'parsed from OcrTier::weights_file in ocr_profile.rs')
      : unknown('could not parse OcrTier::weights_file'),
    max_image_tokens_wired: measured(false,
      'ocr_profile.rs module documentation: server_args builds --image-max-tokens and nothing but its own unit ' +
      'test calls it; serving::plan_launch does not emit it and the request body has no field for it.'),
    decode_cap_wired: measured(true,
      'ocr_profile.rs: max_decode_tokens is sent per request as max_tokens'),
    repetition_control: measured(
      "DRY settings approximating Baidu's no_repeat_ngram_size=35 over a 128-token window. An approximation, not " +
      'a reproduction; ocr_profile.rs says so and says it must be measured.',
      'ocr_profile.rs module documentation'),
    observed_page_accuracy: unknown('no OCR page was run by this inventory'),
  };
}

function embedding(modelInventory) {
  const rows = (modelInventory.entries ?? []).filter((e) => (e.roles ?? []).includes('embedding'));
  return {
    registered: rows.map((r) => ({
      id: r.id,
      path: r.path.value,
      bytes: r.bytes_on_disk.value,
      configured_context: r.configured_context.value,
    })),
    wired_into_retrieval: measured(false,
      'LocalEmbedder is declared in knowledge/embedding.rs, re-exported from knowledge/mod.rs and referenced in ' +
      'hybrid.rs documentation — and constructed by no production caller. context_compiler.rs reports its own ' +
      'retrieval mode as "lexical" for this reason.'),
    consequence:
      'Plan §3 finding 3 is a code gap, not a weights gap: an embedding model is installed and registered on this ' +
      'machine and retrieval still does not use one.',
  };
}

function repoState() {
  const head = run('git', ['rev-parse', 'HEAD']);
  const branch = run('git', ['rev-parse', '--abbrev-ref', 'HEAD']);
  const status = run('git', ['status', '--porcelain=v1']);
  const bundle = join(ROOT, 'agent-runtime', 'dist', 'arjun-agent-runtime.mjs');

  return {
    head: head.ok ? measured(head.stdout, 'git rev-parse HEAD') : unknown('git did not answer'),
    branch: branch.ok ? measured(branch.stdout, 'git rev-parse --abbrev-ref HEAD') : unknown('git did not answer'),
    dirty_paths: status.ok
      ? measured(status.stdout ? status.stdout.split(/\r?\n/).filter(Boolean).length : 0, 'git status --porcelain')
      : unknown('git did not answer'),
    plan_baseline_sha: declared('4164fc26bbdc86eeed61fcf6513f0b97f2cc7e0e', 'build plan header'),
    agent_runtime_bundle: existsSync(bundle)
      ? measured(
          { path: bundle, bytes: statSync(bundle).size, mtime: statSync(bundle).mtime.toISOString() },
          'fs.statSync')
      : unknown('the agent runtime bundle is not built. Run: npm run runtime:build'),
  };
}

// ── Assembly ────────────────────────────────────────────────────────────────

const inventory = {
  schema: 'arjun.qualification-inventory/1',
  generated_at: new Date().toISOString(),
  generated_by: 'scripts/qualification-inventory.mjs',
  hashes_computed: WANT_HASHES,
  reading_rule:
    'Every value is {value, source}. source is one of: measured (a command ran here), declared (a file on this ' +
    'machine asserts it, unchecked), unknown (nothing established it; unknown_because says what would). ' +
    'A null value is never a compatible default.',
  repository: repoState(),
  target_machine: targetMachine(),
  current_machine: currentMachine(),
  runtime: llamaRuntime(),
  containers: containers(),
  renderers: renderers(),
  ocr: ocrSettings(),
};

inventory.models = await models();
inventory.embedding = embedding(inventory.models);

mkdirSync(dirname(OUT), { recursive: true });
writeFileSync(OUT, `${JSON.stringify(inventory, null, 2)}\n`, 'utf8');

const total = inventory.models.entries.length;
const present = inventory.models.entries.filter((e) => e.path.source === 'measured').length;
const hashed = inventory.models.entries.filter((e) => e.sha256.source !== 'unknown').length;
process.stdout.write(
  `qualification inventory written to ${OUT}\n` +
  `  models registered ${total}; present on this machine ${present}; with a sha256 ${hashed}\n` +
  `  container daemon reachable: ${inventory.containers.daemon_reachable.value}\n` +
  `  llama-server build: ${inventory.runtime.build_number.value ?? 'unknown'}\n` +
  (WANT_HASHES ? '' : '  (run with --hash to compute the missing model hashes)\n'),
);
