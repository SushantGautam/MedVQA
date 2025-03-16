import gradio as gr
from datetime import datetime, timedelta
import json

# Sample data structure to hold submission information
submissions = [
    {"user": "User1", "task": "task1", "submitted_time": datetime.now() -
     timedelta(hours=1)},
    {"user": "User2", "task": "task2", "submitted_time": datetime.now() -
     timedelta(days=1)},
    {"user": "User3", "task": "task1", "submitted_time": datetime.now() -
     timedelta(minutes=30)},
    {"user": "User1", "task": "task1", "submitted_time": datetime.now() -
     timedelta(hours=1)},
    {"user": "User2", "task": "task2", "submitted_time": datetime.now() -
     timedelta(days=1)},
    {"user": "User3", "task": "task1", "submitted_time": datetime.now() -
     timedelta(minutes=30)},
    {"user": "User1", "task": "task1", "submitted_time": datetime.now() -
     timedelta(hours=1)},
    {"user": "User2", "task": "task2", "submitted_time": datetime.now() -
     timedelta(days=1)},
    {"user": "User3", "task": "task1", "submitted_time": datetime.now() -
     timedelta(minutes=30)},
    {"user": "User2", "task": "task2", "submitted_time": datetime.now() -
     timedelta(days=1)},
    {"user": "User3", "task": "task1", "submitted_time": datetime.now() -
     timedelta(minutes=30)},
    {"user": "User1", "task": "task1", "submitted_time": datetime.now() -
     timedelta(hours=1)},
    {"user": "User2", "task": "task2", "submitted_time": datetime.now() -
     timedelta(days=1)},
    {"user": "User3", "task": "task1", "submitted_time": datetime.now() -
     timedelta(minutes=30)},
    {"user": "User2", "task": "task2", "submitted_time": datetime.now() -
     timedelta(days=1)},
    {"user": "User3", "task": "task1", "submitted_time": datetime.now() -
     timedelta(minutes=30)},
    {"user": "User1", "task": "task1", "submitted_time": datetime.now() -
     timedelta(hours=1)},
    {"user": "User2", "task": "task2", "submitted_time": datetime.now() -
     timedelta(days=1)},
    {"user": "User3", "task": "task1", "submitted_time": datetime.now() -
     timedelta(minutes=30)},
         {"user": "User2", "task": "task2", "submitted_time": datetime.now() -
     timedelta(days=1)},
    {"user": "User3", "task": "task1", "submitted_time": datetime.now() -
     timedelta(minutes=30)},
    {"user": "User1", "task": "task1", "submitted_time": datetime.now() -
     timedelta(hours=1)},
    {"user": "User2", "task": "task2", "submitted_time": datetime.now() -
     timedelta(days=1)},
    {"user": "User3", "task": "task1", "submitted_time": datetime.now() -
     timedelta(minutes=30)},
         {"user": "User2", "task": "task2", "submitted_time": datetime.now() -
     timedelta(days=1)},
    {"user": "User3", "task": "task1", "submitted_time": datetime.now() -
     timedelta(minutes=30)},
    {"user": "User1", "task": "task1", "submitted_time": datetime.now() -
     timedelta(hours=1)},
    {"user": "User2", "task": "task2", "submitted_time": datetime.now() -
     timedelta(days=1)},
    {"user": "User3", "task": "task1", "submitted_time": datetime.now() -
     timedelta(minutes=30)},
         {"user": "User2", "task": "task2", "submitted_time": datetime.now() -
     timedelta(days=1)},
    {"user": "User3", "task": "task1", "submitted_time": datetime.now() -
     timedelta(minutes=30)},
    {"user": "User1", "task": "task1", "submitted_time": datetime.now() -
     timedelta(hours=1)},
    {"user": "User2", "task": "task2", "submitted_time": datetime.now() -
     timedelta(days=1)},
    {"user": "User3", "task": "task1", "submitted_time": datetime.now() -
     timedelta(minutes=30)},
         {"user": "User2", "task": "task2", "submitted_time": datetime.now() -
     timedelta(days=1)},
    {"user": "User3", "task": "task1", "submitted_time": datetime.now() -
     timedelta(minutes=30)},
    {"user": "User1", "task": "task1", "submitted_time": datetime.now() -
     timedelta(hours=1)},
    {"user": "User2", "task": "task2", "submitted_time": datetime.now() -
     timedelta(days=1)},
    {"user": "User3", "task": "task1", "submitted_time": datetime.now() -
     timedelta(minutes=30)},
    # ... add more sample data as needed ...
]


def time_ago(submitted_time):
    delta = datetime.now() - submitted_time
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
        new_submissions = json.load(file)
        for submission in new_submissions:
            submission["submitted_time"] = datetime.strptime(
                submission["submitted_time"], "%Y-%m-%d %H:%M:%S")
            submissions.append(submission)
        return "Submissions added successfully!"
    except Exception as e:
        return f"Error: {str(e)}"


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
            outputs=output_table,
            title="ImageCLEFmed-MEDVQA-GI-2025 Submissions",
            description="Filter and search submissions by task type and user."
        )
    with gr.Tab("Upload Submission", visible=False):
        gr.Interface(
            api_name="UploadSubmission",
            fn=add_submission,
            inputs=upload_button,
            outputs="text",
            title="Upload Submissions",
            description="Upload a JSON file to add new submissions."
        )

demo.launch()
