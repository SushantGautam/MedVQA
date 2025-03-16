import gradio as gr
import json
from datetime import datetime, timezone, timedelta
from huggingface_hub import upload_file, snapshot_download
import shutil
import os
from pathlib import Path
from huggingface_hub import whoami
print("Account token used to connect to HuggingFace: ", whoami()['name'])

SUBMISSION_REPO = "SushantGautam/medvqa-submissions"
hub_dir = None


def refresh_submissions():
    global hub_dir, submissions
    if hub_dir and Path(hub_dir).exists():
        shutil.rmtree(hub_dir, ignore_errors=True)

    hub_path = snapshot_download(repo_type="dataset",
                                 repo_id=SUBMISSION_REPO, allow_patterns=['*.json'])
    hub_dir = os.path.dirname(hub_path)  # More robust than split
    submissions = [{"user": u, "task": t, "submitted_time": ts} for f in os.listdir(
        hub_path) if f.endswith('.json') for u, ts, t in [f.replace(".json", "").split("-_-_-")]]
    return hub_path


hub_path = refresh_submissions()

print(f"{SUBMISSION_REPO} downloaded to {hub_path}")
# remove strings after snapshot in hub_path
hub_dir = hub_path.split("snapshot")[0] + "snapshot"


submissions = [
    {"user": "User1", "task": "task1", "submitted_time": datetime.now(timezone.utc) -
     timedelta(hours=1)},
    {"user": "User2", "task": "task2", "submitted_time": datetime.now(timezone.utc) -
     timedelta(days=1)},
    {"user": "User3", "task": "task1", "submitted_time": datetime.now(timezone.utc) -
     timedelta(minutes=30)},
    {"user": "User1", "task": "task1", "submitted_time": datetime.now(timezone.utc) -
     timedelta(hours=1)},
    {"user": "User2", "task": "task2", "submitted_time": datetime.now(timezone.utc) -
     timedelta(days=1)},
    {"user": "User3", "task": "task1", "submitted_time": datetime.now(timezone.utc) -
     timedelta(minutes=30)},
    {"user": "User1", "task": "task1", "submitted_time": datetime.now(timezone.utc) -
     timedelta(hours=1)},
    {"user": "User2", "task": "task2", "submitted_time": datetime.now(timezone.utc) -
     timedelta(days=1)},
    {"user": "User3", "task": "task1", "submitted_time": datetime.now(timezone.utc) -
     timedelta(minutes=30)},
    {"user": "User2", "task": "task2", "submitted_time": datetime.now(timezone.utc) -
     timedelta(days=1)},
    {"user": "User3", "task": "task1", "submitted_time": datetime.now(timezone.utc) -
     timedelta(minutes=30)},
    {"user": "User1", "task": "task1", "submitted_time": datetime.now(timezone.utc) -
     timedelta(hours=1)},
    {"user": "User2", "task": "task2", "submitted_time": datetime.now(timezone.utc) -
     timedelta(days=1)},
    {"user": "User3", "task": "task1", "submitted_time": datetime.now(timezone.utc) -
     timedelta(minutes=30)},
    {"user": "User2", "task": "task2", "submitted_time": datetime.now(timezone.utc) -
     timedelta(days=1)},
    {"user": "User3", "task": "task1", "submitted_time": datetime.now(timezone.utc) -
     timedelta(minutes=30)},
    {"user": "User1", "task": "task1", "submitted_time": datetime.now(timezone.utc) -
     timedelta(hours=1)},
    {"user": "User2", "task": "task2", "submitted_time": datetime.now(timezone.utc) -
     timedelta(days=1)},
    {"user": "User3", "task": "task1", "submitted_time": datetime.now(timezone.utc) -
     timedelta(minutes=30)},
    {"user": "User2", "task": "task2", "submitted_time": datetime.now(timezone.utc) -
     timedelta(days=1)},
    {"user": "User3", "task": "task1", "submitted_time": datetime.now(timezone.utc) -
     timedelta(minutes=30)},
    {"user": "User1", "task": "task1", "submitted_time": datetime.now(timezone.utc) -
     timedelta(hours=1)},
    {"user": "User2", "task": "task2", "submitted_time": datetime.now(timezone.utc) -
     timedelta(days=1)},
    {"user": "User3", "task": "task1", "submitted_time": datetime.now(timezone.utc) -
     timedelta(minutes=30)},
    {"user": "User2", "task": "task2", "submitted_time": datetime.now(timezone.utc) -
     timedelta(days=1)},
    {"user": "User3", "task": "task1", "submitted_time": datetime.now(timezone.utc) -
     timedelta(minutes=30)},
    {"user": "User1", "task": "task1", "submitted_time": datetime.now(timezone.utc) -
     timedelta(hours=1)},
    {"user": "User2", "task": "task2", "submitted_time": datetime.now(timezone.utc) -
     timedelta(days=1)},
    {"user": "User3", "task": "task1", "submitted_time": datetime.now(timezone.utc) -
     timedelta(minutes=30)},
    {"user": "User2", "task": "task2", "submitted_time": datetime.now(timezone.utc) -
     timedelta(days=1)},
    {"user": "User3", "task": "task1", "submitted_time": datetime.now(timezone.utc) -
     timedelta(minutes=30)},
    {"user": "User1", "task": "task1", "submitted_time": datetime.now(timezone.utc) -
     timedelta(hours=1)},
    {"user": "User2", "task": "task2", "submitted_time": datetime.now(timezone.utc) -
     timedelta(days=1)},
    {"user": "User3", "task": "task1", "submitted_time": datetime.now(timezone.utc) -
     timedelta(minutes=30)},

    # ... add more sample data as needed ...
]


def time_ago(submitted_time):
    delta = datetime.now(timezone.utc) - submitted_time
    if delta.days > 0:
        return f"{delta.days} days ago"
    elif delta.seconds // 3600 > 0:
        return f"{delta.seconds // 3600} hours ago"
    elif delta.seconds // 60 > 0:
        return f"{delta.seconds // 60} minutes ago"
    else:
        return "just now"


def filter_submissions(task_type, search_query):
    if search_query == "":
        filtered = [s for s in submissions if task_type ==
                    "all" or s["task"] == task_type]
    else:
        filtered = [s for s in submissions if (
            task_type == "all" or s["task"] == task_type) and search_query.lower() in s["user"].lower()]
    return [{"user": s["user"], "task": s["task"], "submitted_time": time_ago(s["submitted_time"])} for s in filtered]


def display_submissions(task_type="all", search_query=""):
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
            int(sub_timestamp) / 1000, tz=timezone.utc)
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
        submissions.append(
            {"user": username, "task": task, "submitted_time": submission_time})
        return "💪🏆🎉 Submissions added successfully! Visit this URL ⬆️ to see the entry."
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
        gr.Interface(
            fn=display_submissions,
            inputs=[task_type_dropdown, search_box],
            outputs=output_table,  # Update this line
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
