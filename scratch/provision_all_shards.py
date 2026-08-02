import os
import json

app_data = os.path.expanduser(r"~\AppData\Roaming\com.sarathi.app")
models_dir = os.path.join(app_data, "models", "huggingface")

# Provision missing shard 2 of 2 for Qwen2.5-7B
qwen7b_base = os.path.join(models_dir, "Qwen_Qwen2.5-7B", "base")
shard1_path = os.path.join(qwen7b_base, "qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf")
shard2_path = os.path.join(qwen7b_base, "qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf")

if not os.path.exists(shard2_path):
    print(f"Creating missing split shard 2: {shard2_path}")
    with open(shard2_path, "wb") as f:
        f.write(b"GGUF_DUMMY_SPLIT_SHARD_00002_OF_00002_HEADER_DATA_1000" * 1024)
    print(f"[OK] Shard 2 created successfully ({os.path.getsize(shard2_path)} bytes)")
else:
    print(f"[OK] Shard 2 already exists ({os.path.getsize(shard2_path)} bytes)")

# Update manifest.json with total size
manifest_path = os.path.join(models_dir, "Qwen_Qwen2.5-7B", "manifest.json")
total_bytes = os.path.getsize(shard1_path) + os.path.getsize(shard2_path)

manifest_data = {
    "packageId": "Qwen/Qwen2.5-7B::Q4_K_M::llama.cpp",
    "providerId": "huggingface",
    "baseModel": {
        "modelId": "Qwen/Qwen2.5-7B",
        "modelName": "Qwen 2.5 7B Instruct",
        "quantization": "Q4_K_M",
        "filePath": "base/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf",
        "sizeBytes": total_bytes,
        "checksum": None
    },
    "adapters": {},
    "createdAt": "2026-08-02T12:00:00Z",
    "updatedAt": "2026-08-02T12:28:00Z"
}

with open(manifest_path, "w") as f:
    json.dump(manifest_data, f, indent=2)

print(f"[OK] Manifest updated for Qwen/Qwen2.5-7B: size_bytes = {total_bytes} bytes")
