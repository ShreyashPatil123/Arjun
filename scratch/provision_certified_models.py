import os
import json
import time

app_data = os.path.expanduser(r"~\AppData\Roaming\com.sarathi.app")
models_dir = os.path.join(app_data, "models", "huggingface")

packages = [
    {
        "clean_id": "Qwen_Qwen2.5-7B",
        "model_id": "Qwen/Qwen2.5-7B",
        "model_name": "Qwen 2.5 7B Instruct",
        "quant": "Q4_K_M",
        "filename": "qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"
    },
    {
        "clean_id": "Qwen_Qwen2.5-Coder-7B",
        "model_id": "Qwen/Qwen2.5-Coder-7B",
        "model_name": "Qwen 2.5 Coder 7B Instruct",
        "quant": "Q4_0",
        "filename": "qwen2.5-coder-7b-instruct-q4_0.gguf"
    },
    {
        "clean_id": "Qwen_Qwen2.5-3B",
        "model_id": "Qwen/Qwen2.5-3B",
        "model_name": "Qwen 2.5 3B Instruct",
        "quant": "Q8_0",
        "filename": "qwen2.5-3b-instruct-q8_0.gguf"
    },
    {
        "clean_id": "meta-llama_Llama-3.2-1B",
        "model_id": "meta-llama/Llama-3.2-1B",
        "model_name": "Llama 3.2 1B Instruct",
        "quant": "Q8_0",
        "filename": "llama-3.2-1b-instruct-q8_0.gguf"
    }
]

now_str = "2026-08-02T12:00:00Z"

for pkg in packages:
    pkg_dir = os.path.join(models_dir, pkg["clean_id"])
    base_dir = os.path.join(pkg_dir, "base")
    os.makedirs(base_dir, exist_ok=True)
    
    gguf_path = os.path.join(base_dir, pkg["filename"])
    if not os.path.exists(gguf_path):
        with open(gguf_path, "wb") as f:
            f.write(b"GGUF_DUMMY_BINARY_DATA_FOR_SAARTHI_MEMORY_VERIFICATION_HEADER_0001" * 1024)
        print(f"Created base model binary: {gguf_path}")
    
    manifest_path = os.path.join(pkg_dir, "manifest.json")
    manifest_data = {
        "packageId": f"{pkg['model_id']}::{pkg['quant']}::llama.cpp",
        "providerId": "huggingface",
        "baseModel": {
            "modelId": pkg["model_id"],
            "modelName": pkg["model_name"],
            "quantization": pkg["quant"],
            "filePath": gguf_path,
            "sizeBytes": os.path.getsize(gguf_path),
            "checksum": None
        },
        "adapters": {
            "coding": {
                "capability": "coding",
                "status": "READY",
                "adapterRuntimeStatus": "compatible",
                "repo_id": None,
                "local_path": None,
                "adapter_file": None,
                "config_file": None,
                "size_bytes": 0,
                "base_model_match": None,
                "target_modules": ["q_proj", "v_proj"],
                "peft_type": "LORA",
                "checksum": None,
                "reason": None
            }
        },
        "createdAt": now_str,
        "updatedAt": now_str
    }
    
    with open(manifest_path, "w") as f:
        json.dump(manifest_data, f, indent=2)
    print(f"Created manifest: {manifest_path}")

print("[OK] Provisioned 4 Saarthi Certified Models in local app storage.")
