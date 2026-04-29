# ⚖️ Legal Document Analyser
**MSc Thesis Project** — AI-Powered Legal Analysis using RAG + LangChain + Claude

---

## Architecture

```
PDF Document
     │
     ▼
PyPDFLoader → RecursiveCharacterTextSplitter (chunks)
                          │
                          ▼
              HuggingFace Embeddings (free, local)
                          │
                          ▼
                   ChromaDB (vector store)
                          │
              ┌───────────┘
              │  Retriever (top-k similarity search)
              │
              ▼
     LCEL RAG Chain:
       {context: retrieved chunks} + {question}
              │
              ▼
     Legal Prompt Template
              │
              ▼
     Claude claude-opus-4-5 (Anthropic API)
              │
              ▼
     Structured Legal Analysis
```

---

## Setup

### 1. Get your Anthropic API key
Sign up at [console.anthropic.com](https://console.anthropic.com) and create an API key.

### 2. Set the environment variable
```bash
export ANTHROPIC_API_KEY='your-key-here'
```
Add this to your `~/.zshrc` or `~/.bash_profile` to persist it.

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## Usage

### Interactive mode (recommended)
```bash
python analyser.py --pdf path/to/contract.pdf
```
This launches an interactive Q&A session with 8 quick legal query shortcuts.

### Single query mode
```bash
python analyser.py --pdf path/to/contract.pdf --query "What are the termination clauses?"
```

---

## Quick Query Shortcuts (in interactive mode)

Type a number 1–8 for pre-built legal queries:

1. What are the key parties involved?
2. What are the termination clauses or exit conditions?
3. What are the payment terms and financial obligations?
4. Are there any penalty or liability clauses?
5. What confidentiality or non-disclosure obligations exist?
6. What is the governing law and jurisdiction?
7. Are there any auto-renewal or notice period requirements?
8. What are the intellectual property ownership terms?

---

## Key Design Decisions (for thesis write-up)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| LLM | Claude claude-opus-4-5 | Superior legal reasoning; structured outputs |
| Embeddings | `all-MiniLM-L6-v2` | Free, local, no API cost; strong semantic similarity |
| Vector Store | ChromaDB | Free, local, persistent across sessions |
| Temperature | `0` | Deterministic — critical for legal accuracy |
| Chunk size | 1000 chars / 200 overlap | Preserves clause context across boundaries |
| Retriever k | 4 | Balances context richness vs. prompt size |
| Prompt design | Source-grounded + risk flagging | Supports AI transparency goals |

---

## Transparency Features
- Every response cites page and excerpt number
- ⚠️ flags ambiguous or high-risk clauses
- Model explicitly says when information is absent (no hallucination)
- Temperature=0 ensures reproducible outputs 

---

## Extending This Project

**Add a web UI** — Wrap with FastAPI + a simple HTML frontend

**Multi-document analysis** — Ingest multiple PDFs into the same ChromaDB collection

**Clause classification** — Add a classification layer to tag clause types automatically

**Confidence scoring** — Use logprobs or a second LLM call to rate answer confidence

**Evaluation** — Use LangSmith or RAGAS to evaluate retrieval and generation quality