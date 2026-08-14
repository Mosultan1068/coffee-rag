# Coffee RAG Assistant

A hybrid AI system combining structured relational data and Retrieval-Augmented Generation (RAG), backed by Supabase, with an MCP server for Claude Desktop and a Gradio web front end. Built as a hands-on learning project to understand RAG, vector databases, MCP, and fine-tuning end to end — including several real, non-obvious bugs found and fixed along the way.

---

## What it does

Ask free-text questions about coffee roasting and brewing, answered by an LLM (Claude) grounded strictly in a curated knowledge base — or browse structured, relational coffee data (origins, varieties, roast profiles, brewing methods) via simple dropdowns.

**Example:**
> "What's the caffeine difference between Arabica and Robusta?"
>
> *"Robusta beans contain roughly 2.2% caffeine by weight, compared to about 1.2% for Arabica — close to double the caffeine content..."*

If a question falls outside the knowledge base, the system says so rather than guessing — a deliberate design choice explained below.

---

## Architecture

Two distinct data types, deliberately kept side by side rather than merged:

- **Structured data** (traditional, normalized): `origins`, `varieties`, `roast_profiles`, `brewing_methods`, `flavor_notes`, and a `variety_flavor_notes` junction table — exact, relational lookups.
- **RAG data**: `raw_documents` (a staging table simulating an upstream data feed) → `documents` / `document_chunks` (chunked, embedded text, searchable by semantic similarity via `pgvector`).

Three layers, kept separate on purpose:
1. **Data layer** — Supabase, accessed via its API
2. **Application logic** — Python: embedding, retrieval, generation
3. **Entry points** — MCP (for Claude Desktop, via LLM tool-calling) and Gradio (direct function calls, deterministic)

**Why two entry points matter:** MCP tool invocation depends on the LLM recognizing a question is relevant — it is *not* guaranteed on every message (confirmed directly: the same question worked when phrased as "according to my knowledge base," but silently skipped the tool and answered from general knowledge when phrased generically). Gradio bypasses that uncertainty entirely by calling the logic directly — a genuine, tested architectural trade-off between conversational flexibility and deterministic reliability.

---

## Tech stack

Python 3.11 · Supabase (Postgres + `pgvector`) · OpenAI (`text-embedding-3-small`) · Anthropic Claude (generation) · MCP Python SDK · Gradio · Docker · GitHub Actions

---

## Grounding strategy: strict, by design

The generation prompt explicitly instructs the model to answer **only** from retrieved context, and a similarity threshold (0.3) gates whether generation is attempted at all — below threshold, the system returns "I don't have information about that" without ever calling the LLM. This was a deliberate choice among three real options (strict / free generation / hybrid), based on asking: *what's the cost of being wrong here, and who does it affect?* For a small, curated knowledge base, strict grounding avoids the risk of confidently wrong answers, at the cost of coverage — a trade-off explicitly considered, not accidental.

---

## Real bugs found and fixed

**1. `ivfflat` index silently missing rows on a small table.** `pgvector`'s approximate-search index only checks a limited number of internal "lists" by default — with only 2 rows spread across 100 lists, one row was never being searched at all. Diagnosed by testing the SQL function directly, then via raw HTTP (bypassing the Python client entirely) to rule out hidden client-side errors. **Root-cause fix:** removed the index entirely — `ivfflat` earns its keep at large scale, not at a handful of rows. A plain exact search is faster and completely reliable at this size.

**2. RLS disabled, flagged by Supabase's own security scanner.** Resolved by enabling RLS with explicit permissive policies — properly configured rather than simply absent, appropriate for a project with no per-user access control yet.

**3. OpenAI's self-serve fine-tuning platform was closed to new organizations mid-project** (a real, current industry change, not a project-specific issue). Pivoted to a local fine-tune using Hugging Face + LoRA instead.

---

## Fine-tuning: an honest result

Fine-tuned `distilgpt2` locally via LoRA (parameter-efficient — only ~0.18% of the model's parameters were actually trained) on 10 Q&A examples. The mechanism worked correctly end to end, but the resulting model's answers were fluent but **not reliably factually grounded** — a direct, concrete illustration of why RAG suits small, precise knowledge bases far better than fine-tuning at this scale. Fine-tuning typically needs far more data (hundreds to thousands of examples) to produce reliable behavior; this result is honest evidence of that, not a failure of the exercise.

---

## Running it

**Locally:**
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
```
Opens automatically at `http://localhost:7860`.

**In Docker:**
```bash
docker build -t coffee-rag-app .
docker run -p 7860:7860 --env-file .env coffee-rag-app
```

**As an MCP server** (for Claude Desktop): add `coffee_server.py` to Claude Desktop's local MCP config, pointing at this project's `venv` Python interpreter.

**Ingesting new content:** add rows to `raw_documents`, then run `python ingest.py` — chunking and embedding happen automatically.

---

## CI/CD

Every push to `main` triggers a GitHub Actions workflow: install dependencies → validate code syntax → build the Docker image. Unlike a prior project's pipeline, this one deliberately does **not** test against live data, since doing so would require exposing real credentials (Supabase, OpenAI, Anthropic keys) to the CI environment — a security boundary kept intentionally. A production pipeline would use GitHub Secrets or mocked API calls to test more deeply without that exposure.

---

## What I'd do differently at scale

- Add GitHub Secrets so CI could run genuine integration tests
- Expand the fine-tuning dataset substantially, and compare against a stronger, instruction-tuned base model
- Add a hybrid grounding mode as a documented alternative to strict grounding, for a less compliance-sensitive use case
- Move ingestion from a manual script run to a real triggered pipeline (e.g. on a schedule, or via a webhook)