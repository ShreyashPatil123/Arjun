/**
 * Bundles the runtime into one file the Rust core can spawn.
 *
 * ## Why bundle at all
 *
 * The vendored packages resolve `@openclaw/*` through tsconfig `paths`, which
 * Node knows nothing about, and they ship TypeScript rather than the `dist/`
 * their manifests advertise (see `vendor/README.md` for why we do not vendor
 * their build). Bundling resolves both in one step and produces a single
 * artifact to hash, sign and ship — which is what the offline installer needs
 * anyway.
 *
 * ## Why the resolution table is read, not written
 *
 * The alias map comes from `vendor/openclaw/tsconfig.json`. Restating it here
 * would create a second source of truth that drifts silently the moment a
 * vendored subpath moves; reading it means the bundler and the type-checker
 * cannot disagree. `vitest.config.ts` reads the same table for the same reason.
 */

import { build } from "esbuild";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const VENDOR = join(ROOT, "vendor/openclaw");
const OUT = join(ROOT, "dist/arjun-agent-runtime.mjs");

const PATHS = JSON.parse(readFileSync(join(VENDOR, "tsconfig.json"), "utf8")).compilerOptions.paths;
const EXTENSIONS = ["", ".ts", ".mts", ".tsx", "/index.ts", "/index.mts"];

function firstExisting(candidate) {
  const stripped = candidate.replace(/\.(js|mjs)$/, "");
  for (const ext of EXTENSIONS) {
    if (existsSync(stripped + ext)) return stripped + ext;
    if (existsSync(candidate + ext)) return candidate + ext;
  }
  return undefined;
}

function resolveFromPaths(id) {
  const exact = PATHS[id];
  if (exact) {
    for (const target of exact) {
      const hit = firstExisting(join(VENDOR, target));
      if (hit) return hit;
    }
  }
  for (const [pattern, targets] of Object.entries(PATHS)) {
    if (!pattern.endsWith("/*")) continue;
    const prefix = pattern.slice(0, -1);
    if (!id.startsWith(prefix)) continue;
    const rest = id.slice(prefix.length);
    for (const target of targets) {
      const hit = firstExisting(join(VENDOR, target.replace("*", rest)));
      if (hit) return hit;
    }
  }
  return undefined;
}

const openclawResolver = {
  name: "arjun:openclaw-source-resolver",
  setup(pluginBuild) {
    pluginBuild.onResolve({ filter: /^@openclaw\// }, (args) => {
      const path = resolveFromPaths(args.path);
      if (!path) {
        // Loud rather than silently external: an unresolved @openclaw import
        // would become a bare require at runtime and fail on the operator's
        // machine instead of on this one.
        throw new Error(`No vendored source for ${args.path}. Check vendor/openclaw/tsconfig.json paths.`);
      }
      return { path };
    });
  },
};

mkdirSync(dirname(OUT), { recursive: true });

const result = await build({
  entryPoints: [join(ROOT, "src/main.ts")],
  outfile: OUT,
  bundle: true,
  platform: "node",
  // Matches package.json engines. Node 22 is the floor the Tauri sidecar ships.
  target: "node22",
  format: "esm",
  // Kept: a stack trace from an operator's machine is the only debugging signal
  // available on an air-gapped deployment, and a minified one is worthless.
  minify: false,
  sourcemap: "linked",
  plugins: [openclawResolver],
  // Node built-ins stay external; everything else is inlined so the shipped
  // artifact does not depend on a node_modules tree being present.
  external: ["node:*"],
  logLevel: "info",
  metafile: true,
});

const bytes = readFileSync(OUT);
const digest = createHash("sha256").update(bytes).digest("hex");
writeFileSync(`${OUT}.sha256`, `${digest}  ${"arjun-agent-runtime.mjs"}\n`);

const inputs = Object.keys(result.metafile.inputs).length;
console.log(`bundled ${inputs} modules -> dist/arjun-agent-runtime.mjs (${(bytes.length / 1024).toFixed(0)} KiB)`);
console.log(`sha256 ${digest}`);
