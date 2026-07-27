# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

A RAG-based legal document analyser (MSc thesis / Northumbria University Enterprise Edge project). Users upload a PDF contract/agreement and ask questions about it; the system retrieves relevant chunks and generates a structured, source-grounded legal analysis (never legal advice).

There are two independent entry points sharing the same RAG pattern but NOT sharing code — changes to the pipeline (prompt, chunking, retrieval params) typically need to be made in both places:
- `app.py` — Streamlit web UI (primary interface)
- `analyser.py` — standalone CLI (`ingest_pdf` → `build_vector_store` → `build_rag_chain` → interactive or single-query loop)

## LLM backend

Both `app.py` and `analyser.py` use `ChatOpenAI` from `langchain-openai`, but pointed at **Groq's** OpenAI-compatible endpoint (`base_url="https://api.groq.com/openai/v1"`) with `model="llama-3.3-70b-versatile"` — not OpenAI itself, and not Claude/Anthropic despite the project name. The API key is read from a generic `API_KEY` env var (not `OPENAI_API_KEY`/`ANTHROPIC_API_KEY`), loaded via `python-dotenv`'s `load_dotenv()` from a local, gitignored `.env` file (see `.env.example` for the template), with a `st.secrets` fallback in `app.py` for Streamlit Cloud. Never write a real key into any tracked file — only `.env` (gitignored).

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Set required API key locally (get a free one at console.groq.com/keys)
cp .env.example .env   # then edit .env and set API_KEY=...

# Run the Streamlit web app
streamlit run app.py

# Run the CLI — interactive mode
python analyser.py --pdf path/to/contract.pdf

# Run the CLI — single query mode
python analyser.py --pdf path/to/contract.pdf --query "What are the termination clauses?"
```

There is no test suite, linter, or build step configured in this repo.

## Architecture (RAG pipeline)

Both entry points follow the same LCEL (LangChain Expression Language) pipeline:

```
PDF → PyPDFLoader → RecursiveCharacterTextSplitter (chunk_size=1000, overlap=200)
    → HuggingFaceEmbeddings (all-MiniLM-L6-v2, local/free)
    → Chroma vector store → retriever (similarity, k=4)
    → format_docs (adds "[Excerpt N — Page P]" citations)
    → LEGAL_PROMPT (ChatPromptTemplate)
    → ChatOpenAI pointed at Groq (llama-3.3-70b-versatile, temperature=0)
    → StrOutputParser → response string
```

Key differences between the two entry points:
- `app.py`'s `build_chain` is `@st.cache_resource`-wrapped, takes uploaded file bytes, writes them to a temp file, builds an **in-memory** (non-persistent) Chroma store, and deletes the temp file after indexing.
- `analyser.py`'s `build_vector_store` persists Chroma to `./chroma_legal_db` (gitignored) so re-runs against the same PDF don't require re-embedding.
- Each file defines its own copy of `LEGAL_PROMPT` and the 8 quick-query shortcuts — they are similar but not identical in wording; keep them in sync intentionally, don't assume editing one updates the other.

`temperature=0` is a deliberate design choice for reproducibility in legal analysis — preserve it if touching LLM config.

The prompt enforces: answer only from retrieved excerpts, cite page numbers, flag ambiguous/high-risk clauses with ⚠️, explicitly say when information is absent, and never give legal advice (analysis only). Preserve these constraints when editing `LEGAL_PROMPT` in either file.

## Deployment

Configured for Streamlit Cloud (`.streamlit/config.toml` sets a light theme). Secrets (`API_KEY`) are expected via `st.secrets` in that environment (set under the app's Settings → Secrets in the dashboard), falling back to the `API_KEY` env var / `.env` file locally. `.streamlit/secrets.toml` and `.env` are both gitignored — real keys never go into a tracked file.
