import { readFileSync, readdirSync, statSync } from 'node:fs';
import { extname, join, relative, resolve } from 'node:path';

const root = resolve(import.meta.dirname, '..');
const sourceRoots = ['src', 'src-tauri/src', 'sidecars'];
const sourceFiles = ['src-tauri/Cargo.toml'];
const textExtensions = new Set([
  '.css', '.html', '.json', '.md', '.py', '.rs', '.toml', '.ts', '.tsx',
]);
const forbiddenContent = [
  { label: 'LoRA capability', pattern: /lora/i },
  { label: 'PEFT adapter support', pattern: /peft/i },
  { label: 'legacy adapter manager', pattern: /adapter_manager/i },
  { label: 'legacy runtime binding', pattern: /lora_binding/i },
  { label: 'legacy frontend adapter service', pattern: /adapters\.service/i },
  { label: 'legacy frontend LoRA service', pattern: /lora\.service/i },
];
const forbiddenPaths = [
  /(^|\/)lora(\/|\.|$)/i,
  /(^|\/)adapter_manager(\/|$)/i,
  /(^|\/)adapters\.service\.ts$/i,
];

function walk(path) {
  return readdirSync(path).flatMap((name) => {
    const child = join(path, name);
    return statSync(child).isDirectory() ? walk(child) : [child];
  });
}

const files = [
  ...sourceRoots.flatMap((directory) => walk(join(root, directory))),
  ...sourceFiles.map((file) => join(root, file)),
].filter((file) => textExtensions.has(extname(file)));

const failures = [];

for (const file of files) {
  const projectPath = relative(root, file).replaceAll('\\', '/');
  for (const pattern of forbiddenPaths) {
    if (pattern.test(projectPath)) {
      failures.push(`${projectPath}: forbidden legacy path`);
    }
  }

  const lines = readFileSync(file, 'utf8').split(/\r?\n/);
  lines.forEach((line, index) => {
    for (const rule of forbiddenContent) {
      if (rule.pattern.test(line)) {
        failures.push(`${projectPath}:${index + 1}: ${rule.label}`);
      }
    }
  });
}

if (failures.length > 0) {
  console.error('LoRA removal gate failed:');
  failures.forEach((failure) => console.error(`- ${failure}`));
  process.exit(1);
}

console.log(`LoRA removal gate passed across ${files.length} product files.`);
