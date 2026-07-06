"""Streamlit frontend for the Enterprise Advanced RAG Dashboard.

Runs the full RAG pipeline in-process (no backend API): Docling parsing,
parent-child chunking, OpenAI embeddings, Qdrant retrieval, FlashRank
re-ranking, and OpenAI answer generation all happen inline in this script.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.chunker.parent_child import ParentChildChunker  # noqa: E402
from src.parser.docling_parser import DoclingParser  # noqa: E402
from src.retriever.reranker import FlashRankReranker  # noqa: E402
from src.retriever.vector_store import QdrantVectorStore  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()

EMBEDDING_MODEL = "text-embedding-3-small"
GENERATION_MODEL = "gpt-4o-mini"
COLLECTION_NAME = "rag_session"
CANDIDATE_TOP_K = 20

RAW_DIR = Path("data/raw")


@st.cache_resource(show_spinner=False)
def get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to your .env file.")
    return OpenAI(api_key=api_key)


@st.cache_resource(show_spinner=False)
def get_parser() -> DoclingParser:
    return DoclingParser()


@st.cache_resource(show_spinner=False)
def get_chunker() -> ParentChildChunker:
    return ParentChildChunker()


@st.cache_resource(show_spinner=False)
def get_reranker() -> FlashRankReranker:
    return FlashRankReranker()


def get_vector_store() -> QdrantVectorStore:
    if "vector_store" not in st.session_state:
        store = QdrantVectorStore(prefer_local=True)
        store.initialize_collection(COLLECTION_NAME)
        st.session_state.vector_store = store
    return st.session_state.vector_store


def embed_texts(client: OpenAI, texts: List[str]) -> List[List[float]]:
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in response.data]


def render_sidebar() -> None:
    with st.sidebar:
        st.header("Pipeline Configuration")
        st.markdown(
            f"""
            - **Parser:** Docling
            - **Chunker:** Parent-Child
            - **Database:** Qdrant (in-memory)
            - **Embeddings:** OpenAI `{EMBEDDING_MODEL}`
            - **Generation:** OpenAI `{GENERATION_MODEL}`
            - **Re-ranker:** FlashRank
            """
        )
        st.divider()
        st.caption("Running fully in-process — no backend API required.")
        num_docs = len(st.session_state.get("ingested_files", []))
        st.caption(f"Documents ingested this session: {num_docs}")


def ingest_document(uploaded_file: Any) -> Dict[str, Any]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    local_path = RAW_DIR / uploaded_file.name
    local_path.write_bytes(uploaded_file.getvalue())

    parser = get_parser()
    chunker = get_chunker()
    client = get_openai_client()
    vector_store = get_vector_store()

    logger.info("Parsing '%s' with DoclingParser", local_path)
    markdown_text = parser.parse_or_load(str(local_path))

    nodes = chunker.create_nodes(markdown_text)

    parent_texts: Dict[str, str] = {
        node.node_id: node.text for node in nodes if "parent_id" not in node.metadata
    }
    child_nodes = [node for node in nodes if "parent_id" in node.metadata]

    if not child_nodes:
        raise ValueError("Chunking produced no child nodes to index.")

    embeddings = embed_texts(client, [node.text for node in child_nodes])

    child_chunks = [
        {
            "node_id": node.node_id,
            "text": node.text,
            "embedding": embedding,
            "parent_id": node.metadata["parent_id"],
            "parent_text": parent_texts.get(node.metadata["parent_id"], node.text),
        }
        for node, embedding in zip(child_nodes, embeddings)
    ]

    vector_store.upsert_child_chunks(COLLECTION_NAME, child_chunks)

    return {
        "file_name": uploaded_file.name,
        "file_size_bytes": local_path.stat().st_size,
        "parent_chunks": len(parent_texts),
        "child_chunks": len(child_chunks),
        "preview": markdown_text[:300],
    }


def render_ingestion_section() -> None:
    st.subheader("Document Ingestion")
    uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

    if uploaded_file is None:
        return

    if uploaded_file.name in st.session_state.get("ingested_files", []):
        st.info(f"'{uploaded_file.name}' has already been ingested this session.")
        return

    try:
        with st.spinner(f"Parsing, chunking, and embedding '{uploaded_file.name}'..."):
            result = ingest_document(uploaded_file)

        st.session_state.setdefault("ingested_files", []).append(uploaded_file.name)
        st.success(
            f"'{uploaded_file.name}' ingested locally: {result['parent_chunks']} parent "
            f"chunk(s), {result['child_chunks']} child chunk(s) indexed."
        )
        with st.expander("Markdown preview"):
            st.text(result["preview"])
    except Exception as exc:
        logger.exception("Failed to process uploaded file '%s'", uploaded_file.name)
        st.error(f"Failed to process '{uploaded_file.name}': {exc}")


def render_sources(sources: List[Dict[str, Any]]) -> None:
    for idx, source in enumerate(sources, start=1):
        section = source.get("section", "Unknown section")
        score = source.get("score")
        score_label = f"{score:.4f}" if isinstance(score, (int, float)) else "N/A"

        with st.expander(f"Source {idx} — {section} (score: {score_label})"):
            st.markdown(f"**Section:** {section}")
            st.markdown(f"**Similarity score:** {score_label}")
            st.markdown("**Source text:**")
            st.text(source.get("text", ""))


def generate_answer(client: OpenAI, query: str, context_chunks: List[Dict[str, Any]]) -> str:
    context_block = "\n\n".join(
        f"[Source {idx}]\n{chunk['parent_text']}" for idx, chunk in enumerate(context_chunks, start=1)
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant answering questions using only the provided "
                "context sources. Cite sources as [Source N]. If the context does not "
                "contain the answer, say so explicitly."
            ),
        },
        {"role": "user", "content": f"Context:\n{context_block}\n\nQuestion: {query}"},
    ]
    response = client.chat.completions.create(model=GENERATION_MODEL, messages=messages, temperature=0.2)
    return response.choices[0].message.content or "No answer returned."


def run_query(query: str) -> Dict[str, Any]:
    client = get_openai_client()
    vector_store = get_vector_store()
    reranker = get_reranker()

    query_embedding = embed_texts(client, [query])[0]
    candidates = vector_store.search(COLLECTION_NAME, query_embedding, top_k=CANDIDATE_TOP_K)

    if not candidates:
        return {"answer": "No documents have been ingested yet — upload a PDF first.", "sources": []}

    reranked = reranker.rerank(query, candidates)

    seen_parents = set()
    deduped = []
    for chunk in reranked:
        parent_id = chunk.get("parent_id")
        if parent_id in seen_parents:
            continue
        seen_parents.add(parent_id)
        deduped.append(chunk)

    answer = generate_answer(client, query, deduped)
    sources = [
        {
            "section": f"Parent chunk {chunk.get('parent_id', 'unknown')[:8]}",
            "score": chunk.get("rerank_score"),
            "text": chunk.get("parent_text", ""),
        }
        for chunk in deduped
    ]
    return {"answer": answer, "sources": sources}


def render_chat_section() -> None:
    st.subheader("Ask a Question")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                render_sources(message["sources"])

    prompt = st.chat_input("Ask a question about your ingested documents...")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Retrieving answer..."):
                payload = run_query(prompt)

            answer = payload["answer"]
            sources = payload["sources"]

            st.markdown(answer)
            render_sources(sources)

            st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
        except Exception as exc:
            logger.exception("Unexpected error while handling query")
            error_message = f"Unexpected error while processing your query: {exc}"
            st.error(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message})


def main() -> None:
    st.set_page_config(page_title="Enterprise Advanced RAG Dashboard", layout="wide")
    st.title("Enterprise Advanced RAG Dashboard")

    render_sidebar()
    render_ingestion_section()
    st.divider()
    render_chat_section()


if __name__ == "__main__":
    main()
