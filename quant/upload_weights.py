"""Upload the exported model weights to a Hugging Face repo so the deployed site can fetch them.
One-time setup, then one command:

    1. Create a (free) HF account, then a token with "write" scope: https://huggingface.co/settings/tokens
    2. quant/.venv/Scripts/python.exe -m pip install -U huggingface_hub
    3. set HF_TOKEN=hf_...        (or `huggingface-cli login`)
    4. quant/.venv/Scripts/python.exe quant/upload_weights.py

Prints the base URL to paste into WEIGHTS_REMOTE in web/machine.mjs (already set if the repo id below matches).
"""
import os, sys
from huggingface_hub import HfApi

REPO_ID = "blazingphoenix7/bitcrush-bc06"   # <user>/<repo> on huggingface.co
FILES = ["qwen3.bin", "manifest.json"]
SRC = os.path.join("web", "weights-qwen3")

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

def main():
    api = HfApi()
    api.create_repo(REPO_ID, repo_type="model", exist_ok=True)
    for f in FILES:
        p = os.path.join(SRC, f)
        print(f"uploading {p} ({os.path.getsize(p)/1e6:.0f} MB) ...")
        api.upload_file(path_or_fileobj=p, path_in_repo=f, repo_id=REPO_ID, repo_type="model")
    base = f"https://huggingface.co/{REPO_ID}/resolve/main/"
    print("\ndone. weights base URL:")
    print("  " + base)
    print("web/machine.mjs WEIGHTS_REMOTE must match this exactly.")

if __name__ == "__main__":
    main()
