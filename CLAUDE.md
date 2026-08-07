# Sarathi

Tauri 2 desktop app for running local LLMs. React 19 + TypeScript + Vite frontend
(`src/`), Rust backend (`src-tauri/`), Python sidecars (`sidecars/`).

Build variants: `npm run tauri:dev:gpu` (CUDA), `npm run tauri:dev:vulkan` (Vulkan),
or `npm run dev:auto` to let `scripts/select-backend.mjs` pick.

## Skill routing

- Changes to the Rust launcher/model logic (`src-tauri/src/launcher/`) → `/review` before merge
- UI or design changes (`src/`) → `/design-review`
- Verifying the app actually runs → `/qa`
- Model download / hardware sizing bugs → `/investigate`

## gstack (REQUIRED — global install)

**Before doing ANY work, verify gstack is installed:**

```bash
test -d ~/.claude/skills/gstack/bin && echo "GSTACK_OK" || echo "GSTACK_MISSING"
```

If GSTACK_MISSING: STOP. Do not proceed. Tell the user:

> gstack is required for all AI-assisted work in this repo.
> Install it:
> ```bash
> git clone --depth 1 https://github.com/garrytan/gstack.git ~/.claude/skills/gstack
> cd ~/.claude/skills/gstack && ./setup --team
> ```
> Then restart your AI coding tool.

Do not skip skills, ignore gstack errors, or work around missing gstack.

Using gstack skills: After install, skills like /qa, /ship, /review, /investigate,
and /browse are available. Use /browse for all web browsing.
Use ~/.claude/skills/gstack/... for gstack file paths (the global path).
