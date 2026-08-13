"""
Connection test — proves the Python <-> Supabase plumbing works,
before any embedding or RAG logic is added.
"""

import os
from dotenv import load_dotenv
from supabase import create_client

# Load SUPABASE_URL and SUPABASE_KEY from the .env file
load_dotenv()

url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_KEY")

if not url or not key:
    raise RuntimeError(
        "Missing SUPABASE_URL or SUPABASE_KEY. "
        "Check that your .env file exists and both values are set."
    )

# Create the Supabase client
supabase = create_client(url, key)

# Read back everything currently in raw_documents
response = supabase.table("raw_documents").select("*").execute()

print(f"Connected successfully. Found {len(response.data)} row(s) in raw_documents:\n")

for row in response.data:
    print(f"- id: {row['id']}")
    print(f"  source_system: {row['source_system']}")
    print(f"  processed: {row['processed']}")
    print(f"  content preview: {row['raw_content'][:80]}...")
    print()