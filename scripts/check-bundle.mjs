#!/usr/bin/env node
/**
 * Bundle gate — evidence about what actually ships, not about what was written.
 *
 * The other gates reason about source. `check-egress.mjs` says there is one
 * place in this repository that constructs an outbound client;
 * `agent-runtime/scripts/audit-vendor.mjs` says the vendored copy of OpenClaw
 * has had its cloud providers deleted. Both are claims about a source tree.
 *
 * This one inspects `agent-runtime/dist/arjun-agent-runtime.mjs`: one file
 * holding the agent loop and every module it genuinely imports, produced by the
 * bundler from a thousand sources. If a provider survived the pruning — through
 * a re-export, a lazy import, a dependency nobody read — it is in that file
 * whatever the source tree says. A reviewer can be handed the artifact and this
 * gate's output together.
 *
 * ## What it can and cannot prove
 *
 * It **can** prove which protocol adapters are registered, that the loopback
 * refusal shipped, and that no channel or web-search module is present. Those
 * are structural facts about the artifact.
 *
 * It **cannot** prove a vendor hostname is absent, because the OpenAI-compatible
 * transport legitimately contains vendor hostnames — as *comparisons*. Code like
 * `baseUrl.includes("api.together.xyz")` exists to detect which flavour of
 * OpenAI-compatible server it is talking to. Deleting those would mean patching
 * a file we otherwise take unchanged, for no gain: the string is a comparison,
 * not a destination.
 *
 * So vendor hostnames are handled as a **reviewed ledger** rather than a ban.
 * Each is recorded with its count and why it is acceptable. A new occurrence, or
 * more of an existing one, fails the gate and has to be looked at by a person.
 * That is the honest version of the check: the surface is small, enumerated, and
 * cannot grow quietly.
 */

import { existsSync, readFileSync, statSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = join(fileURLToPath(new URL('.', import.meta.url)), '..');
/**
 * The artifact under inspection.
 *
 * Overridable by argument so the gate can be pointed at a deliberately doctored
 * copy and shown to fail. A gate nobody has watched fail is a gate nobody knows
 * works — see `scripts/check-bundle.test.mjs`.
 */
const BUNDLE =
  process.argv[2] ?? join(ROOT, 'agent-runtime', 'dist', 'arjun-agent-runtime.mjs');

/** The only protocol adapter this build may register. */
const PERMITTED_APIS = new Set(['openai-completions']);

/**
 * How the bundler renders a built-in adapter registration.
 *
 * Matched against the bundled shape rather than the source shape: esbuild
 * rewrites `registerApiProvider({ api: "x", ... })` into a shorthand whose
 * literal lives at the `createLazyRegistration` call site instead. A matcher
 * written against the source silently matches nothing here — which is why the
 * integrity check below treats "no matches" as a failure.
 */
const ADAPTER_REGISTRATION = /createLazyRegistration\(\s*"([a-z0-9-]+)"/g;

/**
 * Strings whose presence means an excluded capability is in the artifact.
 *
 * Each is specific enough that it cannot appear innocently: a channel's API
 * host, a web-search vendor's endpoint, the cloud code-execution tool's name.
 */
const EXCLUDED_MARKERS = [
  ['graph.facebook.com', 'WhatsApp channel'],
  ['api.telegram.org', 'Telegram channel'],
  ['discord.com/api', 'Discord channel'],
  ['slack.com/api', 'Slack channel'],
  ['api.search.brave.com', 'Brave web search'],
  ['api.firecrawl.dev', 'Firecrawl web fetch'],
  ['api.tavily.com', 'Tavily web search'],
  ['api.exa.ai', 'Exa web search'],
  ['duckduckgo.com', 'DuckDuckGo web search'],
  ['code_execution', 'xAI-hosted code execution'],
  ['openshell', 'remote shell backend'],
  ['ClawHub', 'plugin installation'],
];

/**
 * Defences that must be present in the shipped artifact.
 *
 * A gate that only checks for absence passes just as happily on an empty file.
 * These assert the protections actually shipped.
 */
const REQUIRED_DEFENCES = [
  ['is not loopback', 'the runtime refuses a non-loopback model endpoint'],
  ['No authorisation grant for', 'a tool cannot execute without a gateway grant'],
  ['tool.authorize', 'every tool call is put to the Rust gateway'],
];

/**
 * Vendor hostnames known to be in the artifact, and why each is acceptable.
 *
 * The count is part of the contract. If a host appears more often than recorded,
 * something new is carrying it and a person has to look. Reviewed against the
 * bundle on 2026-08-26.
 */
const REVIEWED_VENDOR_HOSTS = new Map([
  [
    'api.openai.com',
    {
      count: 4,
      why:
        "the openai SDK's default baseURL fallback and its JSDoc. ARJUN always sets " +
        'model.baseUrl explicitly and refuses anything non-loopback before a client is built ' +
        '(agent-runtime/src/run.ts, src-tauri/src/serving/probe.rs), so the default is never ' +
        'reached — but it is a real fallback and is recorded here rather than argued away.',
    },
  ],
  [
    'api.together.xyz',
    { count: 2, why: 'a baseUrl.includes() comparison that detects an OpenAI-compatible flavour' },
  ],
  [
    'api.together.ai',
    { count: 2, why: 'the same comparison, alternate domain' },
  ],
  [
    'ai-gateway.vercel.sh',
    { count: 1, why: 'a baseUrl.includes() comparison for a routing quirk' },
  ],
  [
    'gateway.ai.cloudflare.com',
    { count: 1, why: 'a baseUrl.includes() comparison' },
  ],
  [
    'platform.openai.com',
    { count: 34, why: "documentation links inside the openai SDK's error messages" },
  ],
  [
    'help.openai.com',
    { count: 2, why: "documentation links inside the openai SDK's error messages" },
  ],
  [
    'auth.openai.com',
    { count: 1, why: "a documentation link inside the openai SDK's error messages" },
  ],
  [
    'ollama.com',
    { count: 1, why: 'a documentation link in a model-catalogue comment' },
  ],
  [
    'docs.expo.dev',
    { count: 1, why: 'a documentation link in a vendored dependency comment' },
  ],
]);

/** Host-shaped literals that are text rather than destinations. */
const IGNORED_URL_HOSTS = new Set([
  '127.0.0.1',
  'localhost',
  '0.0.0.0',
  '::1',
  'json-schema.org',
  'www.w3.org',
  'schemas.openxmlformats.org',
  'spdx.org',
  'www.apache.org',
  'opensource.org',
  'unlicense.org',
  'creativecommons.org',
  'developer.mozilla.org',
  'tc39.es',
  'nodejs.org',
  'github.com',
  'www.github.com',
  'raw.githubusercontent.com',
  'docs.openclaw.ai',
  'openclaw.ai',
  'example.com',
  'www.example.com',
  'registry.npmjs.org',
]);

const URL_LITERAL = /https?:\/\/([a-z0-9.-]+)/gi;

const findings = [];
const fail = (what, detail) => findings.push({ what, detail });

if (!existsSync(BUNDLE)) {
  console.error(
    `bundle gate: the runtime bundle is missing at ${BUNDLE}.\n` +
      'Build it first:  npm run runtime:build',
  );
  process.exit(2);
}

const source = readFileSync(BUNDLE, 'utf8');
const sizeMiB = (statSync(BUNDLE).size / 1024 / 1024).toFixed(1);

// 1. Exactly the permitted protocol adapters, and the check must still work.
const registered = new Set();
for (const match of source.matchAll(ADAPTER_REGISTRATION)) {
  registered.add(match[1]);
}
if (registered.size === 0) {
  fail(
    'gate integrity',
    'no adapter registration was found, so this scan is no longer verifying anything. The ' +
      'bundler has probably changed how createLazyRegistration is emitted — fix the matcher ' +
      'before trusting a pass.',
  );
}
for (const api of registered) {
  if (!PERMITTED_APIS.has(api)) {
    fail('extra protocol adapter', `${api} is registered in the shipped bundle`);
  }
}

// 2. No excluded capability.
for (const [marker, description] of EXCLUDED_MARKERS) {
  if (source.includes(marker)) {
    fail('excluded capability', `${description} (matched ${JSON.stringify(marker)})`);
  }
}

// 3. The defences shipped.
for (const [marker, description] of REQUIRED_DEFENCES) {
  if (!source.includes(marker)) {
    fail(
      'missing defence',
      `${description} — expected ${JSON.stringify(marker)} in the bundle and it is absent`,
    );
  }
}

// 4. Vendor hostnames match the reviewed ledger exactly.
const seen = new Map();
for (const match of source.matchAll(URL_LITERAL)) {
  const host = match[1].toLowerCase().replace(/[.]+$/, '');
  if (IGNORED_URL_HOSTS.has(host)) continue;
  seen.set(host, (seen.get(host) ?? 0) + 1);
}
for (const [host, count] of seen) {
  const reviewed = REVIEWED_VENDOR_HOSTS.get(host);
  if (!reviewed) {
    fail(
      'unreviewed host',
      `${host} appears ${count} time(s) and is not in the reviewed ledger. Read the context, ` +
        'then either remove what carries it or record it with a reason.',
    );
    continue;
  }
  if (count > reviewed.count) {
    fail(
      'host surface grew',
      `${host} appears ${count} time(s), reviewed at ${reviewed.count}. Something new is ` +
        'carrying it.',
    );
  }
}

if (findings.length === 0) {
  const hosts = [...seen.keys()].length;
  console.log(
    `bundle gate: pass — ${sizeMiB} MiB, adapters: ${[...registered].join(', ')}, ` +
      `${hosts} reviewed vendor hostname(s), all defences present`,
  );
  process.exit(0);
}

console.error(`bundle gate: FAIL — ${findings.length} finding(s) in the shipped runtime\n`);
for (const { what, detail } of findings) {
  console.error(`  [${what}] ${detail}`);
}
console.error(
  '\nThe bundle is what ships. A finding here means the pruning in agent-runtime/vendor/ ' +
    'did not reach something that is still being imported.',
);
process.exit(1);
