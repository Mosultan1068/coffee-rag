"""
Raw HTTP debug script — bypasses supabase-py entirely, talks directly
to Supabase's REST API, to see the exact status code and response body
with nothing hidden by the client library.
"""

import os
import json
import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Get a real embedding for a real question
response = openai_client.embeddings.create(
    model="text-embedding-3-small",
    input="What happens to beans during light roasting?"
)
query_embedding = response.data[0].embedding
print(f"Embedding generated, length: {len(query_embedding)}")

url = f"{SUPABASE_URL}/rest/v1/rpc/match_document_chunks"
headers = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json"
}
payload = {
    "query_embedding": query_embedding,
    "match_count": 3
}

print(f"\nPOSTing to: {url}")
resp = requests.post(url, headers=headers, data=json.dumps(payload))

print(f"\nStatus code: {resp.status_code}")
print(f"Response body: {resp.text}")