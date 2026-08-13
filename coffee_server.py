"""
Coffee RAG MCP Server

Wraps the retrieval + generation pipeline (from generate.py) as a
single MCP tool, so Claude Desktop (or any MCP client) can call it
directly.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from supabase import create_client
from openai import OpenAI
import anthropic

# Load .env from this file's own folder — same fix applied in the
# fault-finding project, since MCP clients launch this script from
# their own working directory, not from C:\coffee_rag.
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)
claude_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

EMBEDDING_MODEL = "text-embedding-3-small"
CLAUDE_MODEL = "claude-sonnet-4-5"
SIMILARITY_THRESHOLD = 0.3

mcp = FastMCP("coffee-rag-server")


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


@mcp.tool()
def ask_coffee_question(question: str) -> str:
    """
    Answer a question about coffee beans and roasting, grounded in a
    curated knowledge base. If the question falls outside that
    knowledge base, says so rather than guessing.

    Args:
        question: Any free-text question about coffee, roasting,
            brewing, or bean origins.

    Returns:
        A generated answer grounded in retrieved context, or a message
        indicating the question is outside the current knowledge base.
    """
    chunks = retrieve_chunks(question)

    if not chunks or chunks[0]["similarity"] < SIMILARITY_THRESHOLD:
        return "I don't have information about that in my coffee knowledge base."

    return generate_answer(question, chunks)


if __name__ == "__main__":
    mcp.run(transport="stdio")
    