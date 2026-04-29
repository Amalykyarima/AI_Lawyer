"""
Legal Document Analyser — Streamlit Frontend
app.py — connects to the RAG backend in analyser.py
"""

import os
import tempfile
import streamlit as st
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Legal Analyser",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

h1, h2, h3 {
    font-family: 'DM Serif Display', serif !important;
}

/* Main header */
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.6rem;
    color: #1a1a2e;
    margin-bottom: 0.2rem;
    line-height: 1.15;
}
.hero-subtitle {
    font-size: 1rem;
    color: #6b7280;
    font-weight: 300;
    margin-bottom: 2rem;
}

/* Analysis response card */
.analysis-card {
    background: #fafaf8;
    border: 1px solid #e5e5e0;
    border-left: 4px solid #1a1a2e;
    border-radius: 8px;
    padding: 1.5rem 1.8rem;
    margin-top: 1rem;
    font-size: 0.95rem;
    line-height: 1.8;
    color: #1f2937;
}

/* Risk warning styling */
.analysis-card p:has(span:contains("⚠️")) {
    background: #fff8ec;
    border-radius: 4px;
    padding: 0.4rem 0.8rem;
}

/* Quick query buttons */
.stButton > button {
    background: #f9f9f7 !important;
    border: 1px solid #d1d5db !important;
    border-radius: 6px !important;
    color: #374151 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.83rem !important;
    font-weight: 400 !important;
    padding: 0.45rem 0.9rem !important;
    text-align: left !important;
    width: 100% !important;
    transition: all 0.15s ease !important;
}
.stButton > button:hover {
    background: #1a1a2e !important;
    border-color: #1a1a2e !important;
    color: #ffffff !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #f4f4f0;
    border-right: 1px solid #e5e5e0;
}
[data-testid="stSidebar"] .stMarkdown h3 {
    font-family: 'DM Serif Display', serif !important;
    font-size: 1.1rem;
    color: #1a1a2e;
}

/* Upload area */
[data-testid="stFileUploader"] {
    border: 2px dashed #d1d5db;
    border-radius: 10px;
    padding: 1rem;
    background: #fafaf8;
}

/* Status badges */
.badge-ready {
    display: inline-block;
    background: #dcfce7;
    color: #166534;
    font-size: 0.75rem;
    font-weight: 500;
    padding: 0.2rem 0.7rem;
    border-radius: 99px;
    margin-left: 0.5rem;
}
.badge-processing {
    display: inline-block;
    background: #fef9c3;
    color: #854d0e;
    font-size: 0.75rem;
    font-weight: 500;
    padding: 0.2rem 0.7rem;
    border-radius: 99px;
    margin-left: 0.5rem;
}

/* Chat history */
.chat-question {
    background: #1a1a2e;
    color: #ffffff;
    border-radius: 8px 8px 2px 8px;
    padding: 0.7rem 1rem;
    margin: 0.8rem 0 0.3rem auto;
    max-width: 80%;
    font-size: 0.9rem;
    width: fit-content;
    margin-left: auto;
}
.chat-answer {
    background: #fafaf8;
    border: 1px solid #e5e5e0;
    border-left: 3px solid #1a1a2e;
    border-radius: 2px 8px 8px 8px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 1rem;
    font-size: 0.88rem;
    line-height: 1.75;
    color: #1f2937;
}

/* Divider */
hr { border-color: #e5e5e0; margin: 1.5rem 0; }
</style>
""", unsafe_allow_html=True)


# ── Legal prompt ───────────────────────────────────────────────────────────────
LEGAL_PROMPT = ChatPromptTemplate.from_template("""
You are an expert legal document analyst. Analyse the document excerpts below and answer the question clearly and precisely.

GUIDELINES:
- Base your answer ONLY on the retrieved excerpts
- Always cite the page number when referencing content
- Flag ambiguous or high-risk clauses with ⚠️
- If the answer is not in the document, say: "This information is not present in the provided document."
- Never provide legal advice — only legal analysis

RETRIEVED EXCERPTS:
{context}

QUESTION:
{question}
""")

QUICK_QUERIES = [
    "What are the key parties involved?",
    "What are the termination clauses?",
    "What are the payment terms and obligations?",
    "Are there any penalty or liability clauses?",
    "What confidentiality obligations exist?",
    "What is the governing law and jurisdiction?",
    "Are there auto-renewal or notice period requirements?",
    "What are the intellectual property ownership terms?",
]


# ── Session state ──────────────────────────────────────────────────────────────
if "chain" not in st.session_state:
    st.session_state.chain = None
if "history" not in st.session_state:
    st.session_state.history = []
if "doc_name" not in st.session_state:
    st.session_state.doc_name = None
if "processing" not in st.session_state:
    st.session_state.processing = False


# ── RAG pipeline ───────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def build_chain(file_bytes: bytes, filename: str):
    """Build the RAG chain from uploaded PDF bytes."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = splitter.split_documents(pages)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
    )

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )

    def format_docs(docs):
        formatted = []
        for i, doc in enumerate(docs, 1):
            page = doc.metadata.get("page", "?")
            formatted.append(f"[Excerpt {i} — Page {page + 1}]\n{doc.page_content}")
        return "\n\n---\n\n".join(formatted)

    api_key = os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")

    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        max_tokens=2048,
        api_key=api_key,
    )

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | LEGAL_PROMPT
        | llm
        | StrOutputParser()
    )

    os.unlink(tmp_path)
    return chain, len(pages), len(chunks)


# ── Layout ─────────────────────────────────────────────────────────────────────
# Sidebar
with st.sidebar:
    st.markdown("### ⚖️ Legal Analyser")
    st.markdown("---")

    uploaded = st.file_uploader(
        "Upload a legal document",
        type=["pdf"],
        help="PDF contracts, agreements, or any legal document",
    )

    if uploaded:
        if uploaded.name != st.session_state.doc_name:
            st.session_state.doc_name = uploaded.name
            st.session_state.history = []
            with st.spinner("Reading and indexing document..."):
                chain, n_pages, n_chunks = build_chain(
                    uploaded.read(), uploaded.name
                )
                st.session_state.chain = chain
            st.success(f"Ready — {n_pages} pages, {n_chunks} chunks indexed")

    st.markdown("---")
    st.markdown("### Quick queries")
    for q in QUICK_QUERIES:
        if st.button(q, key=f"quick_{q}"):
            st.session_state._pending_query = q

    st.markdown("---")
    st.markdown("---")
    st.markdown("""
    <div style="font-size: 0.78rem; color: #6b7280; line-height: 1.7;">
        <div style="font-weight: 500; color: #1a1a2e; margin-bottom: 0.4rem">🔒 Privacy & Data Notice</div>
        <p>Documents you upload are processed <strong>in memory only</strong> and are never stored, saved, or logged on any server.</p>
        <p>Your files are <strong>not used for AI training</strong>, not shared with third parties, and not retained after your session ends.</p>
        <p>Queries are sent to the OpenAI API solely to generate your analysis. OpenAI's data handling is governed by their <a href="https://openai.com/policies/api-data-usage-policies" target="_blank" style="color:#1a1a2e">API data usage policy</a>.</p>
        <p>This tool provides <strong>legal analysis only</strong> — not legal advice. Always consult a qualified solicitor for legal decisions.</p>
        <hr style="border-color:#e5e5e0; margin: 0.6rem 0"/>
        <p style="color:#9ca3af">MSc Thesis Project · AI-Powered Legal Analysis · RAG + LangChain + GPT-4o</p>
    </div>
    """, unsafe_allow_html=True)


# Main area
st.markdown('<p class="hero-title">AI Legal Document Analyser</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">Upload a legal document, then ask questions or use the quick query shortcuts.</p>', unsafe_allow_html=True)

if not st.session_state.chain:
    st.markdown("""
        <div style="
            border: 1.5px dashed #d1d5db;
            border-radius: 10px;
            padding: 2rem;
            text-align: center;
            color: #6b7280;
        ">
            <div style="font-size: 2rem">←</div>
            <div style="font-weight: 500; color: #1a1a2e; margin-bottom: 0.3rem">Upload a document to get started</div>
            <div style="font-size: 0.85rem">Click <strong>Browse files</strong> in the sidebar on the left</div>
        </div>
        """, unsafe_allow_html=True)
else:
    # Chat history
    for item in st.session_state.history:
        st.markdown(f'<div class="chat-question">{item["q"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="chat-answer">{item["a"]}</div>', unsafe_allow_html=True)

    # Handle quick query button press
    pending = st.session_state.pop("_pending_query", None)

    # Input
    with st.form("query_form", clear_on_submit=True):
        col1, col2 = st.columns([5, 1])
        with col1:
            user_input = st.text_input(
                "Ask a question",
                value=pending or "",
                placeholder="e.g. What are the termination conditions?",
                label_visibility="collapsed",
            )
        with col2:
            submitted = st.form_submit_button("Analyse", use_container_width=True)

    if submitted and user_input:
        with st.spinner("Analysing document..."):
            response = st.session_state.chain.invoke(user_input)
        st.session_state.history.append({"q": user_input, "a": response})
        st.rerun()