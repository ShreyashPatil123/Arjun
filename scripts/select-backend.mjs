#!/usr/bin/env node
/**
 * Picks the fastest llama.cpp backend this machine can actually build, then runs
 * the requested tauri command with the matching cargo feature.
 *
 * Backend selection in llama.cpp is a compile-time decision: `llama-cpp-sys-2`
 * links GGML with CUDA or Vulkan support baked in, so a binary built without one
 * cannot acquire it at runtime no matter what hardware it later finds. That is
 * why this lives in the build and not in the app.
 *
 * Nothing here is specific to one machine. Everything is probed:
 *   - CUDA   requires an NVIDIA driver *and* the toolkit (nvcc), not just a GPU.
 *   - Vulkan requires the SDK (glslc/VULKAN_SDK), not just the runtime loader.
 *   - Neither present -> CPU, which always works.
 *
 * Usage:  node scripts/select-backend.mjs <dev|build> [extra tauri args...]
 */

import { spawnSync, spawn } from 'node:child_process';
import { existsSync, readdirSync } from 'node:fs';
import path from 'node:path';

const isWindows = process.platform === 'win32';

/** True when `cmd` runs and exits cleanly. */
function probe(cmd, args) {
  const r = spawnSync(cmd, args, { stdio: 'ignore', shell: isWindows });
  return r.status === 0;
}

function detectBackend() {
  if (process.env.SARATHI_BACKEND) {
    const forced = process.env.SARATHI_BACKEND.toLowerCase();
    return { feature: forced === 'cpu' ? null : forced, why: 'forced by SARATHI_BACKEND' };
  }

  const hasNvidiaGpu = probe('nvidia-smi', ['--query-gpu=name', '--format=csv,noheader']);
  const hasNvcc = probe('nvcc', ['--version']);
  const hasVulkanSdk = Boolean(process.env.VULKAN_SDK) || probe('glslc', ['--version']);

  if (hasNvidiaGpu && hasNvcc) {
    // nvcc existing is not the same as nvcc being able to build here. Each CUDA
    // release pins a range of supported MSVC versions, and a newer Visual Studio
    // is rejected outright — `-allow-unsupported-compiler` exists but NVIDIA's
    // own warning is that it "may cause incorrect run time execution", which is
    // not something to opt into silently for inference kernels.
    const mismatch = cudaHostCompilerMismatch();
    if (!mismatch) {
      return { feature: 'cuda', why: 'NVIDIA GPU + CUDA toolkit (nvcc) present' };
    }

    // Checked before falling through so the reason survives into the log. A
    // silent downgrade here reads as "this machine has no GPU", which sends
    // whoever is debugging it in entirely the wrong direction.
    if (hasVulkanSdk) {
      return { feature: 'vulkan', why: `${mismatch} — using Vulkan on the same GPU instead` };
    }
    return { feature: null, why: `${mismatch}, and no Vulkan SDK to fall back to` };
  }

  if (hasVulkanSdk) {
    return { feature: 'vulkan', why: 'Vulkan SDK present (vendor-neutral GPU offload)' };
  }

  if (hasNvidiaGpu && !hasNvcc) {
    return { feature: null, why: 'NVIDIA GPU found but no CUDA toolkit — install it for GPU offload' };
  }
  return { feature: null, why: 'no GPU build toolchain found' };
}

/**
 * Whether the installed CUDA refuses the installed Visual Studio.
 *
 * Returns an explanation when they are incompatible, or `null` when the pair is
 * fine or cannot be determined. Undeterminable counts as fine: guessing wrong in
 * that direction costs one failed build, while guessing wrong the other way
 * would abandon a working CUDA setup on a machine that has one.
 */
function cudaHostCompilerMismatch() {
  if (!isWindows) return null;

  const nvcc = spawnSync('nvcc', ['--version'], { encoding: 'utf8', shell: true });
  const cuda = /release (\d+)\.(\d+)/.exec(nvcc.stdout ?? '');
  if (!cuda) return null;

  const vcvars = findVcvars();
  if (!vcvars) return null;

  // The Visual Studio major version sits in the install path — "…/2022/…" for
  // the year-named releases, "…/18/…" for the numbered ones that followed.
  const edition = /Microsoft Visual Studio[\\/](\d+)[\\/]/.exec(vcvars);
  if (!edition) return null;

  const vs = Number(edition[1]);
  // CUDA 12.x supports Visual Studio 2017 through 2022. Anything after 2022 is
  // numbered rather than year-named, so a small number means a newer release.
  const supported = vs >= 2017 && vs <= 2022;
  if (supported) return null;

  return `CUDA ${cuda[1]}.${cuda[2]} does not support the installed Visual Studio ${vs}`;
}

/**
 * Locates a Visual Studio environment script.
 *
 * Windows CUDA builds need two things the default path lacks: a host compiler on
 * PATH, and a generator that does not depend on the CUDA MSBuild integration
 * (which the toolkit ships but does not always register). Searching rather than
 * assuming a path keeps this working across VS editions and years.
 */
function findVcvars() {
  const roots = [process.env['ProgramFiles(x86)'], process.env.ProgramFiles].filter(Boolean);

  const vswhere = roots
    .map((r) => path.join(r, 'Microsoft Visual Studio', 'Installer', 'vswhere.exe'))
    .find(existsSync);

  if (vswhere) {
    const r = spawnSync(vswhere, ['-latest', '-products', '*', '-property', 'installationPath'], {
      encoding: 'utf8',
    });
    const base = (r.stdout || '').trim().split(/\r?\n/)[0];
    if (base) {
      const candidate = path.join(base, 'VC', 'Auxiliary', 'Build', 'vcvars64.bat');
      if (existsSync(candidate)) return candidate;
    }
  }

  // vswhere is absent on Build Tools-only installs; walk the standard layout.
  for (const root of roots) {
    const vsRoot = path.join(root, 'Microsoft Visual Studio');
    if (!existsSync(vsRoot)) continue;
    for (const year of readdirSync(vsRoot)) {
      const editions = path.join(vsRoot, year);
      let entries = [];
      try {
        entries = readdirSync(editions);
      } catch {
        continue;
      }
      for (const edition of entries) {
        const candidate = path.join(editions, edition, 'VC', 'Auxiliary', 'Build', 'vcvars64.bat');
        if (existsSync(candidate)) return candidate;
      }
    }
  }
  return null;
}

const [, , mode = 'build', ...rest] = process.argv;
if (!['dev', 'build'].includes(mode)) {
  console.error(`usage: select-backend.mjs <dev|build> [tauri args]`);
  process.exit(2);
}

const { feature, why } = detectBackend();
const tauriArgs = [mode, ...(feature ? ['--features', feature] : []), ...rest];

console.log(`[sarathi] backend: ${feature ?? 'cpu'} — ${why}`);
console.log(`[sarathi] tauri ${tauriArgs.join(' ')}`);

// A CUDA build on Windows needs the MSVC environment and the Ninja generator;
// everywhere else the default toolchain is already correct.
if (isWindows && feature === 'cuda') {
  const vcvars = findVcvars();
  if (!vcvars) {
    console.error('[sarathi] CUDA selected but no Visual Studio C++ environment was found.');
    console.error('[sarathi] Install VS Build Tools with the C++ workload, or set SARATHI_BACKEND=cpu.');
    process.exit(1);
  }
  console.log(`[sarathi] msvc env: ${vcvars}`);

  const quoted = tauriArgs.map((a) => (a.includes(' ') ? `"${a}"` : a)).join(' ');
  const line = `"${vcvars}" >nul && set CMAKE_GENERATOR=Ninja&& npx tauri ${quoted}`;

  // `windowsVerbatimArguments` is required, not optional. Node escapes quotes
  // as \" so that a program using the C runtime's argument parser sees them —
  // but cmd.exe does not use that parser. It reads the raw command line, so the
  // backslashes arrive as literal characters and the whole thing fails with
  // "'\"C:\Program Files\...\vcvars64.bat\"' is not recognized". The flag tells
  // Node to hand the string over untouched, which is what cmd expects.
  const child = spawn('cmd', ['/c', line], {
    stdio: 'inherit',
    windowsVerbatimArguments: true,
  });
  child.on('exit', (code) => process.exit(code ?? 1));
} else {
  const child = spawn('npx', ['tauri', ...tauriArgs], { stdio: 'inherit', shell: isWindows });
  child.on('exit', (code) => process.exit(code ?? 1));
}
