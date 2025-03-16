import gradio as gr
import json
from datetime import datetime, timezone
from huggingface_hub import upload_file, snapshot_download
import shutil
import os
import glob
from pathlib import Path
from huggingface_hub import whoami
print("Account token used to connect to HuggingFace: ", whoami()['name'])

SUBMISSION_REPO = "SushantGautam/medvqa-submissions"
hub_path = None

submissions = None  # [{"user": u, "task": t, "submitted_time": ts}]


def refresh_submissions():
    global hub_path, submissions
    if hub_path and Path(hub_path).exists():
        shutil.rmtree(hub_path, ignore_errors=True)
        print("Deleted existing submissions")

    hub_path = snapshot_download(repo_type="dataset",
                                 repo_id=SUBMISSION_REPO, allow_patterns=['**/*.json'])
    print("Downloaded submissions to: ", hub_path)
    if not os.path.exists(hub_path):
        os.makedirs(hub_path)  # empty repo case
    print("os.listdir(hub_path):", os.listdir(hub_path))
    all_jsons = glob.glob(hub_path + "/**/*.json", recursive=True)
    json_files = [f.split("/")[-1] for f in all_jsons]
    submissions = []
    for file in json_files:
        username, sub_timestamp, task = file.replace(
            ".json", "").split("-_-_-")
        submissions.append({"user": username, "task": task,
                           "submitted_time": sub_timestamp})
    return hub_path


hub_path = refresh_submissions()

print(f"{SUBMISSION_REPO} downloaded to {hub_path}")
# remove strings after snapshot in hub_path
hub_dir = hub_path.split("snapshot")[0] + "snapshot"


def time_ago(submitted_time):
    return str(datetime.fromtimestamp(int(submitted_time), tz=timezone.utc)) + " UTC"


def filter_submissions(task_type, search_query):
    if search_query == "":
        filtered = [s for s in submissions if task_type ==
                    "all" or s["task"] == task_type]
    else:
        filtered = [s for s in submissions if (
            task_type == "all" or s["task"] == task_type) and search_query.lower() in s["user"].lower()]
    return [{"user": s["user"], "task": s["task"], "submitted_time": time_ago(s["submitted_time"])} for s in filtered]


def display_submissions(task_type="all", search_query=""):
    if submissions is None:
        refresh_submissions()
    filtered_submissions = filter_submissions(task_type, search_query)
    return [[s["user"], s["task"], s["submitted_time"]] for s in filtered_submissions]


def add_submission(file):
    try:
        print("Received submission: ", file)
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        username, sub_timestamp, task = file.replace(
            ".json", "").split("-_-_-")
        submission_time = datetime.fromtimestamp(
            int(sub_timestamp), tz=timezone.utc)
        assert task in ["task1", "task2"], "Invalid task type"
        assert len(username) > 0, "Invalid username"
        assert submission_time < datetime.now(
            timezone.utc), "Invalid submission time"
        print("Adding submission...", username, task, submission_time)
        upload_file(
            repo_type="dataset",
            path_or_fileobj=file,
            path_in_repo=task+"/"+file.split("/")[-1],
            repo_id=SUBMISSION_REPO
        )
        refresh_submissions()
        return "💪🏆🎉 Submissions registered successfully to the system!"
    except Exception as e:
        raise Exception(f"Error adding submission: {e}")


def refresh_page():
    return "Pong! Submission server is alive! 😊"


# Define Gradio interface components
output_table = gr.Dataframe(
    headers=["User", "Task", "Submitted Time"], value=display_submissions(), scale=5,)
task_type_dropdown = gr.Dropdown(
    choices=["all", "task1", "task2"], value="all", label="Task Type")
search_box = gr.Textbox(value="", label="Search User")
upload_button = gr.File(label="Upload JSON", file_types=["json"])

# Create a tabbed interface
with gr.Blocks(title="ImageCLEFmed-MEDVQA-GI-2025 Submissions") as demo:
    with gr.Tab("View Submissions"):
        gr.Markdown("### Submissions Table")
        gr.Interface(
            fn=display_submissions,
            inputs=[task_type_dropdown, search_box],
            outputs=output_table,
            title="ImageCLEFmed-MEDVQA-GI-2025 Submissions",
            description="Filter and search submissions by task type and user."
        )
    with gr.Tab("Upload Submission", visible=True):
        file_input = gr.File(label="Upload JSON", file_types=["json"])
        upload_output = gr.Textbox(label="Result")  # Add this line
        file_input.upload(add_submission, file_input,
                          upload_output)

    with gr.Tab("Refresh API", visible=False):
        gr.Interface(
            api_name="RefreshAPI",
            fn=refresh_page,
            inputs=[],
            outputs="text",
            title="Refresh API",
            description="Hidden interface to refresh the API."
        )

demo.launch()
