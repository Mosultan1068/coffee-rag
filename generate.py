"""
Generation script — the full RAG loop:
question -> embed -> retrieve -> threshold gate -> generate answer.

This builds directly on retrieve.py. The only new piece is: when
sufficiently relevant chunks ARE found, they're handed to Claude with
an instruction to answer only from that context. When they're not
found, we skip the LLM call entirely, exactly as in retrieve.py.
"""

import os
from dotenv import load_dotenv
from supabase import create_client
from openai import OpenAI
import anthropic

load_dotenv()

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)
claude_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

EMBEDDING_MODEL = "text-embedding-3-small"
CLAUDE_MODEL = "claude-sonnet-4-5"  # a current, capable, cost-reasonable model for this task
SIMILARITY_THRESHOLD = 0.3


def get_embedding(text: str) -> list[float]:
    response = openai_client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return response.data[0].embedding


def retrieve_chunks(question: str, match_count: int = 3):
    query_embedding = get_embedding(question)
    result = supabase.rpc("match_document_chunks", {
        "query_embedding": query_embedding,
        "match_count": match_count
    }).execute()
    return result.data


def generate_answer(question: str, chunks: list[dict]) -> str:
    """
    Builds a prompt from the retrieved chunks and asks Claude to answer
    strictly using that context — this is the guard against the LLM
    filling gaps with outside knowledge, discussed earlier.
    """
    context = "\n\n".join(f"- {chunk['content']}" for chunk in chunks)

    prompt = f"""You are a coffee roasting assistant. Answer the question
using ONLY the context provided below. Do not use any outside knowledge.
If the context doesn't fully answer the question, say what it does cover
and be clear about what it doesn't.

Context:
{context}

Question: {question}

Answer:"""

    response = claude_client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.content[0].text


def ask(question: str):
    print(f"\nQuestion: {question}")

    chunks = retrieve_chunks(question)

    if not chunks or chunks[0]["similarity"] < SIMILARITY_THRESHOLD:
        print("Answer: I don't have information about that in my coffee knowledge base.")
        return

    answer = generate_answer(question, chunks)
    print(f"Answer: {answer}")


if __name__ == "__main__":
    ask("How does dark roasting affect flavor?")
    ask("What's the exchange rate between USD and GBP today?")
    