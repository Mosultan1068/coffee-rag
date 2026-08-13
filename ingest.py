"""
Ingestion script — the "glue" between raw_documents (the simulated
upstream staging table) and documents/document_chunks (the RAG tables).

What it does, step by step:
1. Reads rows from raw_documents where processed = false
2. Splits each one into chunks (simple paragraph-based chunking)
3. Generates an OpenAI embedding for each chunk
4. Writes a row to `documents` (the parent) and one row per chunk to
   `document_chunks` (with its embedding)
5. Marks the raw_documents row as processed = true, so it won't be
   ingested again next time this script runs
"""

import os
from dotenv import load_dotenv
from supabase import create_client
from openai import OpenAI

load_dotenv()

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

EMBEDDING_MODEL = "text-embedding-3-small"  # produces 1536-dimensional vectors,
                                             # matching the schema we set up


def chunk_text(text: str, max_words: int = 80) -> list[str]:
    """
    Simple chunking: split on sentences, then group sentences together
    until we approach max_words per chunk. Deliberately simple for a
    proof of concept — a production pipeline might use smarter,
    overlap-aware chunking.
    """
    sentences = [s.strip() for s in text.split(".") if s.strip()]
    chunks = []
    current_chunk = []
    current_word_count = 0

    for sentence in sentences:
        word_count = len(sentence.split())
        if current_word_count + word_count > max_words and current_chunk:
            chunks.append(". ".join(current_chunk) + ".")
            current_chunk = []
            current_word_count = 0
        current_chunk.append(sentence)
        current_word_count += word_count

    if current_chunk:
        chunks.append(". ".join(current_chunk) + ".")

    return chunks


def get_embedding(text: str) -> list[float]:
    """Calls OpenAI to generate an embedding for a piece of text."""
    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return response.data[0].embedding


def ingest_raw_documents():
    # Step 1: fetch unprocessed rows
    result = supabase.table("raw_documents").select("*").eq("processed", False).execute()
    raw_rows = result.data

    if not raw_rows:
        print("No unprocessed raw_documents found. Nothing to ingest.")
        return

    print(f"Found {len(raw_rows)} unprocessed row(s) to ingest.\n")

    for raw_row in raw_rows:
        print(f"Processing raw_documents row {raw_row['id']}...")

        # Step 2: create the parent `documents` row.
        # Using the first few words of the raw content as a simple title,
        # since raw_documents doesn't have a title field of its own.
        title_preview = " ".join(raw_row["raw_content"].split()[:6]) + "..."
        doc_insert = supabase.table("documents").insert({
            "title": title_preview,
            "source_url": None
        }).execute()
        document_id = doc_insert.data[0]["id"]

        # Step 3: chunk the raw content
        chunks = chunk_text(raw_row["raw_content"])
        print(f"  Split into {len(chunks)} chunk(s).")

        # Step 4: embed each chunk and insert it
        for index, chunk in enumerate(chunks):
            embedding = get_embedding(chunk)
            supabase.table("document_chunks").insert({
                "document_id": document_id,
                "chunk_index": index,
                "content": chunk,
                "embedding": embedding,
                "metadata": {"source_system": raw_row["source_system"]}
            }).execute()

        # Step 5: mark the raw row as processed
        supabase.table("raw_documents").update({"processed": True}).eq("id", raw_row["id"]).execute()
        print(f"  Marked raw_documents row {raw_row['id']} as processed.\n")

    print("Ingestion complete.")


if __name__ == "__main__":
    ingest_raw_documents()