<div align="center">

# 🪷 Sarathi (सारथी)

### *Universal Local AI Orchestrator, Hardware-Matched LLM Engine & Hybrid Memory Platform*

[![Tauri v2](https://img.shields.io/badge/Tauri-v2.0-blue?style=for-the-badge&logo=tauri&logoColor=white)](https://tauri.app/)
[![Rust](https://img.shields.io/badge/Rust-1.93-orange?style=for-the-badge&logo=rust&logoColor=white)](https://www.rust-lang.org/)
[![React](https://img.shields.io/badge/React-19.0-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://react.dev/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.7-3178C6?style=for-the-badge&logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![Python Sidecar](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

<p align="center">
  <b>Sarathi</b> is an intelligent, hardware-aware local AI desktop orchestrator. It combines real-time physical system profiling, deterministic memory budgeting, parallel LoRA capability adapter discovery, in-process GGUF inference via <code>llama.cpp</code>, and a production-grade <b>Hybrid Local Memory Engine</b>.
</p>

---

</div>

## 🌟 Highlights & Architecture Matrix

```mermaid
graph TD
    subgraph Hardware Telemetry & Scoring
        A[DirectX 12 / DXGI / WMI / Vulkan] --> Profile[Hardware Telemetry]
        Profile --> Scorer[Sarathi Local Memory Scorer]
        HF[Hugging Face Hub API] --> Catalog[GGUF Catalog Provider]
        Catalog --> Scorer
        Scorer --> Categories[Recommended / Compatible / May Run]
    end

    subgraph Native Download & LoRA Pipeline
        Categories --> Downloader[Async Resumable Downloader]
        Downloader --> LoRA[5 Parallel LoRA Capability Handles]
        LoRA --> Registry[Single Source of Truth Manifest]
    end

    subgraph Phase 6 Hybrid Memory Engine
        Chat[User Interface Chat] --> MemMgr[Rust MemoryManager Facade]
        MemMgr --> Stdio[SidecarAdapter: Stdio NDJSON-RPC]
        Stdio -- Zero Sockets -- Sidecar[Python Memory Sidecar]
        Sidecar --> Mem0[Mem0: Dynamic Fact Extraction]
        Sidecar --> Letta[Letta: Working Memory Blocks]
        Sidecar --> Zep[Zep: Temporal Decay & Summaries]
        Sidecar --> LlamaIndex[LlamaIndex: RAG Chunking]
        
        Mem0 --> SQLite[(Single Source of Truth: SQLite sarathi.db)]
        Letta --> SQLite
        Zep --> SQLite
        
        SQLite --> Injector[Prompt Injection Engine]
        Injector --> LLM[Llama.cpp Inference Engine]
    end
```

---

## ✨ System Features

### 🔬 1. Deep Physical System Telemetry
- **Hardware Telemetry**: Scans Windows WMI/CIM, DirectX 12 (DXGI), Vulkan, and System APIs.
- **Memory Domain Separation**: Distinguishes Dedicated Video RAM (VRAM) from Shared System Memory and Physical System RAM.
- **Multi-GPU & iGPU Awareness**: Tailored offloading calculations across NVIDIA CUDA, AMD ROCm/Vulkan, Intel Arc/OneAPI, and CPU-only setups.

### 🧠 2. Hardware-Matched Model Recommendation Engine
- **Live Hugging Face Catalog**: Dynamic retrieval of open-weight GGUF architectures (Qwen, Llama, Gemma, Mistral, Phi, DeepSeek, etc.).
- **Deterministic Memory Budgeting**: Calculates model weight footprint ($W_{\text{bytes}} = \frac{N_{\text{params}} \times \text{bpw}}{8} \times 1.06$) and KV-cache overhead (accounting for MHA/GQA, layer depth, head dimensions, and context windows up to 128k).

### ⚡ 3. Parallel Resumable Downloader & LoRA Adapter Pipeline
- **Concurrent LoRA Capability Discovery**: Spawns 5 independent async `tokio::spawn` tasks for capabilities (`coding`, `reasoning`, `tool-calling`, `mathematics`, `research`) running in parallel with the base model GGUF download.
- **HTTP Range Header Resume**: Automatic resume for interrupted `.part` weight files. Zero zero-byte resets.
- **Single Source of Truth Manifest**: Prevents accidental adapter deletion or re-download loops.

### 🔮 4. In-Process LLM Inference Engine (`llama-cpp-2`)
- **Direct GPU Offloading**: Configurable GPU layer offloading and thread allocation.
- **Dynamic Prompt Formatting**: Built-in Jinja/ChatML, Llama 3, Gemma, and Mistral template formatting.
- **Leak-Proof Stream Parser**: Real-time filtering of special control tokens (`<|im_end|>`, `<|im_start|>`) and reasoning tags (`<think>...</think>`).

### 🪷 5. Phase 6 Production Hybrid Local Memory Engine
- **Direct Framework Integration**: Integrates proven open-source memory systems (**Mem0**, **Letta**, **Zep**, **LlamaIndex**) directly within a plugin-based Python sidecar.
- **Zero-Firewall Stdio IPC**: Operates over newline-delimited JSON-RPC 2.0 via `stdin`/`stdout`. Zero listening ports, zero firewall authorization prompts, 100% child process lifetime coupling.
- **Strict Storage Ownership**: Frameworks act strictly as pure processors. Sarathi owns 100% of storage, transactions, and schema migrations in `sqlite:sarathi.db`.
- **Model-Switch Context Preservation**: User context, user profiles, project memories, and summaries persist across model switches (e.g. Qwen $\rightarrow$ Mistral) and application restarts.

---

## 🛠 Tech Stack

| Component | Technologies Used |
| :--- | :--- |
| **Desktop Core** | [Tauri v2](https://tauri.app/) (Native C++ / Rust Window Manager) |
| **Backend Core** | [Rust 1.93](https://www.rust-lang.org/) (Async Tokio, Rusqlite, Reqwest, Sysinfo, WinAPI, Serde) |
| **Inference Engine** | `llama-cpp-2` (CUDA / Vulkan / CPU native bindings) |
| **Memory Sidecar** | [Python 3.11](https://www.python.org/) (Stdio NDJSON-RPC, Mem0, Letta, Zep, LlamaIndex) |
| **Frontend UI** | [React 19](https://react.dev/), [TypeScript 5.7](https://www.typescriptlang.org/), [Vite](https://vitejs.dev/) |
| **Database** | SQLite (`sqlite:sarathi.db` with Migration V2) |

---

## 🚀 Getting Started

### Prerequisites

- **Node.js**: v18+
- **Rust Toolchain**: 1.75+
- **Python**: 3.10+ (for embedded memory sidecar)
- **Build Tools**: Visual Studio 2022 Build Tools (with C++ and Windows SDK)

### Installation & Local Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/ShreyashPatil123/Sarathi.git
   cd Sarathi
   ```

2. **Install frontend dependencies**:
   ```bash
   npm install
   ```

3. **Run in Development Mode**:
   ```bash
   npm run tauri dev
   ```

4. **Run Unit Tests**:
   ```bash
   cd src-tauri
   cargo test --lib memory_engine::tests
   ```

5. **Build Release Binary**:
   ```bash
   npx tauri build
   ```

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for details.
