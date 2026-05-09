"""
Streamlit frontend for the Multimodal RAG System.

Communicates exclusively with the FastAPI backend via HTTP.
No backend services are imported directly.

Run with:
    streamlit run src/rag/frontend/app.py

Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7
"""

from __future__ import annotations

import streamlit as st

from rag.frontend.api_client import (
    APIError,
    delete_collection,
    ingest_pdf,
    list_collections,
    query_documents,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Multimodal RAG System",
    page_icon="📄",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Sidebar — collection management
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("📚 Collections")

    collections_result = list_collections()

    if isinstance(collections_result, APIError):
        st.error(f"Could not load collections: {collections_result.message}")
        available_collections: list[str] = []
    else:
        available_collections = collections_result

    if available_collections:
        st.write(f"**{len(available_collections)} collection(s):**")
        for col_name in available_collections:
            col1, col2 = st.columns([3, 1])
            col1.write(f"• {col_name}")
            if col2.button("🗑", key=f"del_{col_name}", help=f"Delete '{col_name}'"):
                result = delete_collection(col_name)
                if isinstance(result, APIError):
                    st.error(f"Delete failed: {result.message}")
                else:
                    st.success(f"Deleted '{col_name}'")
                    st.rerun()
    else:
        st.info("No collections yet. Ingest a PDF to create one.")

    st.divider()
    st.caption("Multimodal RAG System v0.1.0")

# ---------------------------------------------------------------------------
# Main area — two tabs
# ---------------------------------------------------------------------------

tab_ingest, tab_query = st.tabs(["📤 Ingest PDF", "🔍 Query"])

# ── Ingest tab ──────────────────────────────────────────────────────────────

with tab_ingest:
    st.header("Ingest a PDF Document")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Upload a PDF to parse, chunk, embed, and store.",
    )

    ingest_collection = st.text_input(
        "Collection name",
        value="default",
        help="Name of the collection to store chunks in.",
    )

    if st.button("Ingest", type="primary", disabled=uploaded_file is None):
        if not ingest_collection.strip():
            st.error("Collection name cannot be empty.")
        else:
            with st.spinner("Ingesting document…"):
                result = ingest_pdf(
                    file_bytes=uploaded_file.read(),
                    filename=uploaded_file.name,
                    collection_name=ingest_collection.strip(),
                )

            if isinstance(result, APIError):
                st.error(f"Ingestion failed: {result.message}")
            else:
                st.success(f"✅ {result.status}")
                col1, col2, col3 = st.columns(3)
                col1.metric("Chunks stored", result.chunk_count)
                col2.metric("Collection", result.collection_name)
                col3.metric("Status", result.status)

# ── Query tab ───────────────────────────────────────────────────────────────

with tab_query:
    st.header("Query the Knowledge Base")

    query_text = st.text_area(
        "Your question",
        placeholder="What does the document say about…?",
        height=100,
    )

    qcol1, qcol2, qcol3 = st.columns(3)

    with qcol1:
        query_collection = st.selectbox(
            "Collection",
            options=available_collections if available_collections else ["default"],
            help="Collection to search.",
        )

    with qcol2:
        top_k = st.number_input(
            "Top-K results",
            min_value=1,
            max_value=20,
            value=5,
            step=1,
            help="Maximum number of chunks to retrieve.",
        )

    with qcol3:
        filter_source = st.text_input(
            "Filter by source (optional)",
            placeholder="filename.pdf",
            help="Restrict retrieval to a specific document.",
        )

    if st.button("Ask", type="primary", disabled=not query_text.strip()):
        with st.spinner("Retrieving and generating answer…"):
            result = query_documents(
                query=query_text.strip(),
                collection_name=query_collection,
                top_k=int(top_k),
                filter_source=filter_source.strip() or None,
            )

        if isinstance(result, APIError):
            st.error(f"Query failed: {result.message}")
        else:
            # Answer
            st.subheader("Answer")
            st.write(result.answer)

            # Citations
            if result.citations:
                st.subheader(f"Citations ({len(result.citations)})")
                citation_data = [
                    {
                        "Source": c.source,
                        "Page": c.page,
                        "Chunk": c.chunk_index,
                        "Score": round(c.score, 4),
                    }
                    for c in result.citations
                ]
                st.dataframe(
                    citation_data,
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("No citations — the answer was generated without retrieved context.")
