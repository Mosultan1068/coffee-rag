"""
Fine-tuning kickoff script.

Uploads the training file to OpenAI and starts a fine-tuning job.
Fine-tuning runs in the background on OpenAI's servers — this script
starts the job and prints its ID; it does NOT wait for it to finish.
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
client = OpenAI(api_key=OPENAI_API_KEY)

TRAINING_FILE_PATH = "coffee_training_data.jsonl"
BASE_MODEL = "gpt-4o-mini-2024-07-18"  # a small, cost-effective model that supports fine-tuning

# Step 1: upload the training file
print("Uploading training file...")
with open(TRAINING_FILE_PATH, "rb") as f:
    uploaded_file = client.files.create(file=f, purpose="fine-tune")

print(f"File uploaded. File ID: {uploaded_file.id}")

# Step 2: start the fine-tuning job
print("\nStarting fine-tuning job...")
job = client.fine_tuning.jobs.create(
    training_file=uploaded_file.id,
    model=BASE_MODEL
)

print(f"\nFine-tuning job started.")
print(f"Job ID: {job.id}")
print(f"Status: {job.status}")
print("\nThis will run in the background on OpenAI's servers.")
print("Run check_finetune_status.py to check on progress.")
