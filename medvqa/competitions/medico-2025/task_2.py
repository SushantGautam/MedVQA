# task_2.py

from gradio_client import Client, handle_file
from huggingface_hub import snapshot_download, login, whoami
import argparse
import os
import subprocess as sp
import time
from datetime import datetime, timezone
import shutil
import json
from huggingface_hub import HfApi, grant_access

HF_GATE_ACESSLIST = ["SushantGautam", "stevenah", "vlbthambawita"]

MEDVQA_SUBMIT = True if os.environ.get('_MEDVQA_SUBMIT_FLAG_', 'FALSE') == 'TRUE' else False
parser = argparse.ArgumentParser(description='Run GI-1015 Task 2 (Clinician-Oriented Multimodal Explanations)')
parser.add_argument('--repo_id', type=str, required=True, help='Path to the HF submission repository')
args, _ = parser.parse_known_args()

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
submission_file = "submission_task2.py"
file_from_validation = "submission_task2.jsonl"  # one object per val_id (1500 lines)

min_library = ["datasets>=3.4.1", "transformers", "evaluate",
               "rouge_score", 'tqdm', "gradio_client>=1.8.0"]

print("🌟 ImageCLEFmed-MEDVQA-GI-2025 🌟", "https://github.com/simula/ImageCLEFmed-MEDVQA-GI-2025")
print("💬 Subtask 2: Clinician-Oriented Multimodal Explanations in GI")
print(f"👀 Analyzing submission repository: {args.repo_id} 👀")

try:
    print(f"Logged in to HuggingFace as: {whoami()['name']}")
except Exception:
    print("⚠️⚠️ Not logged in to HuggingFace! Please get your login token from https://huggingface.co/settings/tokens 🌐")
    login()

client = Client("SimulaMet/Medico-2025")
print("💓 Communicating with the Submission Server: Ping!")
result = client.predict(api_name="/refresh_page")
print(result)

hf_username = whoami()['name']
assert len(hf_username) > 0, "🚫 HuggingFace login failed for some reason"
current_timestamp = int(time.time())

# Download only what we need
snap_dir = snapshot_download(
    repo_id=args.repo_id,
    allow_patterns=[submission_file, file_from_validation]
)

# Basic presence checks
subm_path = os.path.join(snap_dir, submission_file)
jsonl_path = os.path.join(snap_dir, file_from_validation)

if not os.path.isfile(subm_path):
    raise FileNotFoundError(f"Submission file '{submission_file}' not found in the repository!")

if not os.path.isfile(jsonl_path):
    raise FileNotFoundError(f"Required predictions file '{file_from_validation}' not found in the repository!")

# === Validation of submission_task2.jsonl ===
print(f"🧪 Validating '{file_from_validation}' formatting…")
valid_lines = 0
with open(jsonl_path, "r", encoding="utf-8") as f:
    for line_num, line in enumerate(f, start=1):
        stripped = line.strip()
        if not stripped:
            continue  # ignore blank lines
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError as e:
            raise ValueError(f"Line {line_num} is not valid JSON: {e}")

        # Minimal field presence checks (do not enforce full schema here to stay minimal)
        if "val_id" not in obj:
            raise ValueError(f"Line {line_num} missing required key 'val_id'.")

        valid_lines += 1

if valid_lines != 1500:
    raise ValueError(f"❌ '{file_from_validation}' must contain exactly 1500 non-empty JSON lines. Found: {valid_lines}")
print(f"✅ JSONL formatting OK (exactly {valid_lines} lines).")

# NOTE: For Subtask 2 we DO NOT run the submission script to create predictions.
print("ℹ️ Skipping execution of the submission script for Subtask 2 (predictions are pre-generated and included).")

print(f"🎉 Validation checks complete. Snapshot dir: {snap_dir}")

if not MEDVQA_SUBMIT:
    print("\nYou can now run `medvqa validate_and_submit ...` to submit Subtask 2.")
else:
    print("🚀 Preparing for submission 🚀")
    # Upload the JSONL file as the artifact
    file_path_to_upload = os.path.join(snap_dir, f"{hf_username}-_-_-{current_timestamp}-_-_-task2.jsonl")
    shutil.copy(jsonl_path, file_path_to_upload)

    # Make the repo public (but gated) and grant access to organizers
    api = HfApi()
    api.update_repo_visibility(args.repo_id, private=False)   # Make public
    api.update_repo_settings(args.repo_id, gated='manual')    # Enable gated access
    for user in HF_GATE_ACESSLIST:
        try:
            grant_access(args.repo_id, user)  # Grant access
        except Exception as e:
            print(user, ":", e)

    print(
        f'''✅ {args.repo_id} model is now made public, but gated, and is shared with organizers.
You should not make the model private or remove/update it until the competition results are announced.
Feel free to re-submit Subtask 2 if you update the repository file(s).
We will notify you if there are any issues with the submission.
''')

    # Hand over the artifact to the server. Subtask 2 has NO public scores;
    # the server will record it and mark score as {"status": "submitted"}.
    result = client.predict(
        file=handle_file(file_path_to_upload),
        api_name="/add_submission"
    )
    print({
        "User": hf_username,
        "Task": "task2",
        "Submitted_time": str(datetime.fromtimestamp(int(current_timestamp), tz=timezone.utc)) + " UTC",
    })
    print(result)
    print("Visit this URL to see the entry: 👇")
    Client("SimulaMet/Medico-2025")

# Optional challenge-evaluate hook intentionally omitted for Subtask 2 (no public scores).
