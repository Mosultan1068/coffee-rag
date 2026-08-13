"""
Retrieval script — the "ask a question, get matching chunks back" step.
No LLM-generated answer yet, deliberately: this proves retrieval itself
works correctly before adding generation on top (Phase 1 goal).
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

EMBEDDING_MODEL = "text-embedding-3-small"

# Below this similarity score, we treat the result as "not relevant enough"
# rather than returning a weak match. Worth experimenting with this number.
SIMILARITY_THRESHOLD = 0.3


def get_embedding(text: str) -> list[float]:
    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return response.data[0].embedding


def retrieve(question: str, match_count: int = 3):
    print(f"\nQuestion: {question}")

    # Step 1: embed the question
    query_embedding = get_embedding(question)
    print(f"  Embedding generated, length: {len(query_embedding)}")

    # Step 2: call the similarity search function via Supabase RPC
    result = supabase.rpc("match_document_chunks", {
        "query_embedding": query_embedding,
        "match_count": match_count
    }).execute()

    print(f"  Full result object: {result}")
    print(f"  Raw RPC response: {result.data}")

    matches = result.data

    if not matches or matches[0]["similarity"] < SIMILARITY_THRESHOLD:
        print("No sufficiently relevant chunks found (below similarity threshold).")
        print("This question is likely outside the knowledge base.")
        return

    print(f"\nTop {len(matches)} matching chunk(s):\n")
    for match in matches:
        print(f"  Similarity: {match['similarity']:.3f}")
        print(f"  Content: {match['content']}")
        print()


if __name__ == "__main__":
    retrieve("What happens to beans during light roasting?")