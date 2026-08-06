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
  if (hasNvidiaGpu && hasNvcc) {
    return { feature: 'cuda', why: 'NVIDIA GPU + CUDA toolkit (nvcc) present' };
  }

  const hasVulkanSdk = Boolean(process.env.VULKAN_SDK) || probe('glslc', ['--version']);
  if (hasVulkanSdk) {
    return { feature: 'vulkan', why: 'Vulkan SDK present (vendor-neutral GPU offload)' };
  }

  if (hasNvidiaGpu && !hasNvcc) {
    return { feature: null, why: 'NVIDIA GPU found but no CUDA toolkit — install it for GPU offload' };
  }
  return { feature: null, why: 'no GPU build toolchain found' };
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
  const child = spawn('cmd', ['/c', line], { stdio: 'inherit' });
  child.on('exit', (code) => process.exit(code ?? 1));
} else {
  const child = spawn('npx', ['tauri', ...tauriArgs], { stdio: 'inherit', shell: isWindows });
  child.on('exit', (code) => process.exit(code ?? 1));
}
