---
sdk: gradio
sdk_version: 5.24.0
app_file: gradio_interface.py
---
# MedVQA

A CLI tool for MedVQA competition (https://github.com/simula/ImageCLEFmed-MEDVQA-GI-2025).

## Installation

```bash
pip install -U medvqa
```
The library is under heavy development. So, we recommend to always make sure you have the latest version installed.

## Usage
Check [ImageCLEFmed-MEDVQA-GI-2025 competition repo](https://github.com/simula/ImageCLEFmed-MEDVQA-GI-2025#-submission-system) for detailed submission instructions.

```bash
medvqa validate_and_submit --competition=gi-2025 --task=1 --repo_id=...
```
where repo_id is your HuggingFace Model repo id (like SushantGautam/XXModelCheckpoint) with the submission script as required by the competition organizers, for eg, submission_task1.py file for task 1.

Submission for task 2 is not yet implemented. Will be soon live. Stay tuned.
