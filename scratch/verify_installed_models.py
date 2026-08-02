import os

app_data = os.path.expanduser(r"~\AppData\Roaming\com.sarathi.app")
models_dir = os.path.join(app_data, "models")

print("Checking Models Directory at:", models_dir)
if not os.path.exists(models_dir):
    print("Models directory does not exist yet. Creating directory...")
    os.makedirs(models_dir, exist_ok=True)
else:
    for root, dirs, files in os.walk(models_dir):
        for f in files:
            print("Found Model File:", os.path.join(root, f))
