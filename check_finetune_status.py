"""
Fine-tuning status check.

Fine-tuning jobs run in the background and can take anywhere from a
few minutes to longer, depending on OpenAI's queue. Run this script
any time to check on the job's current status.
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# List recent fine-tuning jobs, most recent first
jobs = client.fine_tuning.jobs.list(limit=5)

if not jobs.data:
    print("No fine-tuning jobs found.")
else:
    for job in jobs.data:
        print(f"Job ID: {job.id}")
        print(f"Status: {job.status}")
        print(f"Model: {job.model}")
        if job.fine_tuned_model:
            print(f"Fine-tuned model ID: {job.fine_tuned_model}")
        if job.status == "failed" and job.error:
            print(f"Error: {job.error}")
        print()
