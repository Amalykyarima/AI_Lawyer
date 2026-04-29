"""
Legal Document Analyser
MSc Thesis Project — AI-Powered Legal Analysis using RAG + LangChain + Claude

Architecture:
  PDF → Text Chunks → ChromaDB (vector store) → Retriever → Claude (via Anthropic API)

Usage:
  python analyser.py --pdf path/to/contract.pdf
  python analyser.py --pdf path/to/contract.pdf --query "What are the termination clauses?"
"""

import os
import argparse
from pathlib import Path

# --- LangChain imports ---
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# ─────────────────────────────────────────────
# 1. LEGAL-SPECIFIC PROMPT TEMPLATE
#    Instructs Claude to behave as a legal analyst,
#    cite sources, flag uncertainty, and avoid giving advice.
# ─────────────────────────────────────────────
LEGAL_PROMPT = ChatPromptTemplate.from_template("""
You are an expert legal document analyst. Your role is to analyse legal documents 
and provide clear, structured insights based strictly on the document content provided.

IMPORTANT GUIDELINES:
- Base your analysis ONLY on the retrieved document excerpts below
- Always cite which section or clause your answer refers to
- If the answer is not found in the document, say: "This information is not present in the provided document."
- Flag any ambiguous or potentially high-risk clauses with ⚠️
- Never provide legal advice — only legal analysis and explanation
- Use plain English where possible, but preserve key legal terms

---
RETRIEVED DOCUMENT EXCERPTS:
{context}
---

USER QUESTION:
{question}

Provide a structured, clear response. If relevant, include:
- Direct answer
- Relevant clause/section reference
- Any risks or ambiguities flagged
""")


# ─────────────────────────────────────────────
# 2. PRE-BUILT LEGAL QUERIES
#    Common questions useful for any legal document
# ─────────────────────────────────────────────
LEGAL_QUICK_QUERIES = [
    "What are the key parties involved in this document?",
    "What are the termination clauses or exit conditions?",
    "What are the payment terms and financial obligations?",
    "Are there any penalty or liability clauses?",
    "What confidentiality or non-disclosure obligations exist?",
    "What is the governing law and jurisdiction?",
    "Are there any auto-renewal or notice period requirements?",
    "What are the intellectual property ownership terms?",
]


# ─────────────────────────────────────────────
# 3. DOCUMENT INGESTION PIPELINE
# ─────────────────────────────────────────────
def ingest_pdf(pdf_path: str) -> list:
    """Load and chunk a PDF document for RAG."""
    print(f"\n📄 Loading document: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    print(f"   ✓ Loaded {len(pages)} pages")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,       # characters per chunk
        chunk_overlap=200,     # overlap to preserve context across chunks
        separators=["\n\n", "\n", ".", " "],  # legal docs have lots of newlines
    )
    chunks = splitter.split_documents(pages)
    print(f"   ✓ Split into {len(chunks)} chunks")
    return chunks


# ─────────────────────────────────────────────
# 4. VECTOR STORE SETUP
#    Uses free local HuggingFace embeddings + ChromaDB
#    No paid embedding API needed
# ─────────────────────────────────────────────
def build_vector_store(chunks: list) -> Chroma:
    """Embed chunks and store in a local ChromaDB vector store."""
    print("\n🔍 Building vector store...")
    print("   Downloading/loading embedding model (first run may take a moment)...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",  # lightweight, fast, free
    )

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_legal_db",  # persists between runs
    )
    print("   ✓ Vector store ready")
    return vector_store


# ─────────────────────────────────────────────
# 5. RAG CHAIN ASSEMBLY
#    Same LCEL pattern as the article, adapted for Claude
# ─────────────────────────────────────────────
def build_rag_chain(vector_store: Chroma):
    """Build the RAG chain: retriever → prompt → Claude → output."""

    # Check for API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "\n❌ ANTHROPIC_API_KEY not set.\n"
            "   Get your key at: https://console.anthropic.com\n"
            "   Then run: export ANTHROPIC_API_KEY='your-key-here'\n"
        )

    # Claude as the LLM — swap from Ollama to ChatOpenAI
    llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            max_tokens=2048,
        )

    # Retriever: fetch top 4 most relevant chunks per query
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )

    def format_docs(docs):
        """Format retrieved chunks with page references for transparency."""
        formatted = []
        for i, doc in enumerate(docs, 1):
            page = doc.metadata.get("page", "unknown")
            formatted.append(f"[Excerpt {i} — Page {page + 1}]\n{doc.page_content}")
        return "\n\n---\n\n".join(formatted)

    # LCEL chain — same pipe pattern as the article
    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | LEGAL_PROMPT
        | llm
        | StrOutputParser()
    )

    return chain


# ─────────────────────────────────────────────
# 6. INTERACTIVE CLI
# ─────────────────────────────────────────────
def run_interactive(chain, pdf_name: str):
    """Interactive Q&A loop with quick query shortcuts."""

    print(f"""
╔══════════════════════════════════════════════════════╗
║          ⚖️  Legal Document Analyser                 ║
║          Powered by RAG + LangChain + Claude         ║
╠══════════════════════════════════════════════════════╣
║  Document: {pdf_name[:42]:<42} ║
╚══════════════════════════════════════════════════════╝

Quick queries (type a number 1-8, or ask your own question):
""")

    for i, q in enumerate(LEGAL_QUICK_QUERIES, 1):
        print(f"  {i}. {q}")

    print("\n  Type 'exit' to quit.\n")

    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit"]:
                print("\n⚖️  Analysis session ended.\n")
                break

            # Quick query shortcut
            if user_input.isdigit() and 1 <= int(user_input) <= len(LEGAL_QUICK_QUERIES):
                question = LEGAL_QUICK_QUERIES[int(user_input) - 1]
                print(f"You: {question}")
            else:
                question = user_input

            print("\n⏳ Analysing document...\n")
            response = chain.invoke(question)
            print(f"⚖️  Analysis:\n{response}\n")
            print("─" * 60 + "\n")

        except KeyboardInterrupt:
            print("\n\n⚖️  Session interrupted.\n")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


# ─────────────────────────────────────────────
# 7. ENTRY POINT
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="AI-Powered Legal Document Analyser using RAG + Claude"
    )
    parser.add_argument("--pdf", required=True, help="Path to the legal PDF document")
    parser.add_argument("--query", help="Single query (skips interactive mode)")
    args = parser.parse_args()

    if not Path(args.pdf).exists():
        print(f"❌ File not found: {args.pdf}")
        return

    # Pipeline
    chunks = ingest_pdf(args.pdf)
    vector_store = build_vector_store(chunks)
    chain = build_rag_chain(vector_store)

    pdf_name = Path(args.pdf).name

    if args.query:
        # Single query mode
        print(f"\n⏳ Analysing: {args.query}\n")
        response = chain.invoke(args.query)
        print(f"⚖️  Analysis:\n{response}\n")
    else:
        # Interactive mode
        run_interactive(chain, pdf_name)


if __name__ == "__main__":
    main()
