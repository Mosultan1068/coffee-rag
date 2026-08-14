"""
Coffee Assistant — Gradio front end

Two tabs, matching the two data types in the project:
- "Ask about coffee" — RAG: free-text question -> embed -> retrieve -> generate
- "Browse coffee data" -> structured: dropdown-driven queries against the
  relational tables (origins, varieties, roast_profiles, brewing_methods)

Calls the retrieval/generation logic directly (no MCP involved) — this is
the deterministic entry point discussed earlier, as opposed to Claude
Desktop's LLM-judgment-based tool calling.
"""

import os
import gradio as gr
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
CLAUDE_MODEL = "claude-sonnet-4-5"
SIMILARITY_THRESHOLD = 0.3


# =====================================================
# RAG side (Tab 1)
# =====================================================

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


def ask_rag(question: str):
    if not question.strip():
        return "Please enter a question.", ""

    try:
        chunks = retrieve_chunks(question)
    except Exception as e:
        # Surface the real error instead of silently treating it as
        # "no matches found" — a failed API call and a genuine empty
        # result are very different problems.
        return f"Error during retrieval: {e}", "Retrieval failed — see error above."

    if not chunks or chunks[0]["similarity"] < SIMILARITY_THRESHOLD:
        score_display = f"Best match similarity: {chunks[0]['similarity']:.3f} (below threshold of {SIMILARITY_THRESHOLD})" if chunks else "No matches found."
        return "I don't have information about that in my coffee knowledge base.", score_display

    answer = generate_answer(question, chunks)
    score_display = f"Best match similarity: {chunks[0]['similarity']:.3f} (threshold: {SIMILARITY_THRESHOLD})"
    return answer, score_display


EXAMPLE_QUESTIONS = [
    "How does dark roasting affect flavor?",
    "What happens to beans during light roasting?",
]


# =====================================================
# Structured data side (Tab 2)
# =====================================================

def get_origin_options():
    result = supabase.table("origins").select("id, country, region").execute()
    return [(f"{row['country']} — {row['region']}", row["id"]) for row in result.data]


def get_roast_level_options():
    result = supabase.table("roast_profiles").select("id, roast_level").execute()
    return [(row["roast_level"], row["id"]) for row in result.data]


def get_brewing_method_options():
    result = supabase.table("brewing_methods").select("id, name").execute()
    return [(row["name"], row["id"]) for row in result.data]


def query_origin(origin_id):
    if not origin_id:
        return "Select an origin above."

    origin = supabase.table("origins").select("*").eq("id", origin_id).single().execute().data
    varieties = supabase.table("varieties").select(
        "name, species, description, variety_flavor_notes(flavor_notes(name))"
    ).eq("origin_id", origin_id).execute().data

    lines = [f"**{origin['country']} — {origin['region']}**",
             f"Altitude: {origin['altitude_m']}m",
             f"Climate: {origin['climate_notes']}", ""]

    if not varieties:
        lines.append("_No varieties recorded for this origin yet._")
    else:
        for v in varieties:
            flavors = ", ".join(fn["flavor_notes"]["name"] for fn in v["variety_flavor_notes"]) or "none recorded"
            lines.append(f"**{v['name']}** ({v['species']}) — {v['description']}")
            lines.append(f"Flavor notes: {flavors}")
            lines.append("")

    return "\n".join(lines)


def query_roast_profile(roast_id):
    if not roast_id:
        return "Select a roast level above."
    row = supabase.table("roast_profiles").select("*").eq("id", roast_id).single().execute().data
    return (f"**{row['roast_level']} Roast**\n\n"
            f"Temperature range: {row['temp_range_c']}°C\n"
            f"Duration: {row['duration_minutes']} minutes\n\n"
            f"{row['description']}")


def query_brewing_method(method_id):
    if not method_id:
        return "Select a brewing method above."
    row = supabase.table("brewing_methods").select("*").eq("id", method_id).single().execute().data
    return (f"**{row['name']}**\n\n"
            f"Grind size: {row['grind_size']}\n"
            f"Brew time: {row['brew_time_minutes']} minutes\n\n"
            f"{row['description']}")


# =====================================================
# Gradio UI
# =====================================================

with gr.Blocks(title="Coffee Assistant") as demo:
    gr.Markdown("# ☕ Coffee Assistant")
    gr.Markdown("A hybrid RAG + structured data demo, backed by Supabase.")

    with gr.Tabs():
        with gr.Tab("Ask about coffee (RAG)"):
            gr.Markdown("Ask a free-text question. Answers are grounded strictly in the ingested knowledge base.")
            question_input = gr.Textbox(label="Your question", placeholder="e.g. How does dark roasting affect flavor?")

            with gr.Row():
                for example in EXAMPLE_QUESTIONS:
                    gr.Button(example, size="sm").click(
                        fn=lambda e=example: e, outputs=question_input
                    )

            ask_button = gr.Button("Ask", variant="primary")
            answer_output = gr.Textbox(label="Answer", lines=6)
            score_output = gr.Textbox(label="Retrieval details", lines=1)

            ask_button.click(fn=ask_rag, inputs=question_input, outputs=[answer_output, score_output])

        with gr.Tab("Browse coffee data (structured)"):
            gr.Markdown("Exact lookups against the relational tables — no embeddings involved.")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Origins & Varieties")
                    origin_dropdown = gr.Dropdown(choices=get_origin_options(), label="Select an origin")
                    origin_output = gr.Markdown()
                    origin_dropdown.change(fn=query_origin, inputs=origin_dropdown, outputs=origin_output)

                with gr.Column():
                    gr.Markdown("### Roast Profiles")
                    roast_dropdown = gr.Dropdown(choices=get_roast_level_options(), label="Select a roast level")
                    roast_output = gr.Markdown()
                    roast_dropdown.change(fn=query_roast_profile, inputs=roast_dropdown, outputs=roast_output)

                with gr.Column():
                    gr.Markdown("### Brewing Methods")
                    brewing_dropdown = gr.Dropdown(choices=get_brewing_method_options(), label="Select a brewing method")
                    brewing_output = gr.Markdown()
                    brewing_dropdown.change(fn=query_brewing_method, inputs=brewing_dropdown, outputs=brewing_output)


if __name__ == "__main__":
    demo.launch(inbrowser=True)