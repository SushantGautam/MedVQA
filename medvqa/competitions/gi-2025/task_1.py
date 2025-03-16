from huggingface_hub import snapshot_download, login, whoami
import sys
import argparse
import os
import subprocess as sp


MEDVQA_SUBMIT = True if os.environ.get(
    '_MEDVQA_SUBMIT_FLAG_', 'FALSE') == 'TRUE' else False
parser = argparse.ArgumentParser(description='Run GI-1015 Task 1 (VQA)')
parser.add_argument('--repo_id', type=str, required=True,
                    help='Path to the HF submission repository')
args, _ = parser.parse_known_args()

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
submission_file = "submission_task1.py"
min_library = ["datasets", "transformers", 'tqdm']

print("🌟 ImageCLEFmed-MEDVQA-GI-2025 🌟",
      "https://github.com/simula/ImageCLEFmed-MEDVQA-GI-2025")
print("🔍 Subtask 1: Algorithm Development for Question Interpretation and Response")
print(f"👀 Analyzing submission repository: {args.repo_id} 👀")

try:
    print(f"Logged in to HuggingFace as: {whoami()['name']}")
except Exception:
    print("⚠️⚠️ Not logged in to HuggingFace! Please get your login token from https://huggingface.co/settings/tokens 🌐")
    login()

snap_dir = snapshot_download(
    repo_id=args.repo_id, allow_patterns=[submission_file, "requirements.txt"])

if not os.path.isfile(os.path.join(snap_dir, submission_file)):
    raise FileNotFoundError(
        f"Submission file '{submission_file}' not found in the repository!")

if os.path.isfile(os.path.join(snap_dir, "predictions.json")):
    os.remove(os.path.join(snap_dir, "predictions.json"))

print("📦 Making sure of the minimum requirements to run the script 📦")
sp.run(["python", "-m", "pip", "install", "-q"] + min_library, check=True)

if os.path.isfile(os.path.join(snap_dir, "requirements.txt")):
    print(
        f"📦 Installing requirements from the submission repo: {args.repo_id}/requirements.txt")
    sp.run(["python", "-m", "pip", "install", "-q", "-r",
            f"{snap_dir}/requirements.txt"], cwd=snap_dir, check=True)

sp.run(["python", f"{snap_dir}/{submission_file}"],
       cwd=snap_dir, check=True)
print(
    f"🎉 The submission script ran successfully, the intermediate files are at {snap_dir}")

if not MEDVQA_SUBMIT:
    print("\n You can now run medvqa validate_and_submit .... command to submit the task.")
else:
    print("🚀 Preparing for submission 🚀")
