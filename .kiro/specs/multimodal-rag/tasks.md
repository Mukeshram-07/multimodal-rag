# Implementation Plan: Multimodal RAG System

## Overview

Implement the Multimodal RAG platform incrementally, starting with the core data models and configuration, then building each pipeline component in isolation, wiring them together through the FastAPI backend, adding the Streamlit frontend, and finally exposing the MCP tool layer. Each step is independently testable. Property-based tests are placed close to the component they validate to catch regressions early.

---

## Tasks

- [x] 1. Project scaffolding and configuration
  - Create the directory structure: `src/rag/{ingestion,embedding,storage,retrieval,generation,api,tools,frontend}/`, `tests/{integration}/`, `tests/`
  - Create `pyproject.toml` (or `requirements.txt`) with pinned dependencies: `fastapi==0.111.0`, `uvicorn==0.29.0`, `pydantic==2.7.1`, `pydantic-settings==2.2.1`, `sentence-transformers==2.7.0`, `chromadb==0.5.0`, `pymupdf==1.24.3`, `httpx==0.27.0`, `streamlit==1.35.0`, `hypothesis==6.100.1`, `pytest==8.2.0`, `pytest-asyncio==0.23.6`, `python-dotenv==1.0.1`
  - Create `src/rag/config.py` implementing the `Settings` Pydantic BaseSettings model with all fields from the design (embedding model, chunk size, chunk overlap, top-k, chroma persist dir, LLM settings, log level, app version)
  - Create `src/rag/exceptions.py` defining `IngestionError`, `EmbeddingError`, `RetrievalError`, `GenerationError`, `ConfigurationError`
  - Create `src/rag/models.py` defining all Pydantic data models: `DocumentMetadata`, `Chunk`, `PageContent`, `SearchResult`, `Citation`, `IngestionResult`, `RetrievalResult`, `GeneratedResponse`, `QueryRequest`, `IngestResponse`, `HealthResponse`
  - Create `src/rag/logging_config.py` setting up structured logging with configurable log level from `Settings`
  - Create `.env.example` with all configurable keys and their defaults
  - _Requirements: 9.1, 9.2, 9.3, 9.4, 10.1, 11.1_

- [x] 2. PDF parsing and chunking
  - [x] 2.1 Implement `PDFParser` in `src/rag/ingestion/pdf_parser.py`
    - Use `PyMuPDF` (`fitz`) to open a PDF and extract per-page text
    - Return `list[PageContent]` with 1-indexed page numbers
    - Raise `IngestionError` with filename and reason on parse failure
    - _Requirements: 1.2, 1.5_

  - [x] 2.2 Write unit tests for `PDFParser`
    - Test with a known small PDF fixture: verify page count and text content
    - Test with a non-PDF file: verify `IngestionError` is raised with descriptive message
    - _Requirements: 1.2, 1.5_

  - [x] 2.3 Implement `Chunker` in `src/rag/ingestion/chunker.py`
    - Accept `chunk_size` and `chunk_overlap` as constructor parameters
    - Split page text into overlapping character-level chunks
    - Attach `DocumentMetadata(source, page, chunk_index)` to each `Chunk`
    - Return empty list (not error) when a page has no text
    - _Requirements: 1.3, 1.4, 1.6_

  - [x] 2.4 Write property test for `Chunker` — chunk metadata integrity
    - **Property 1: Chunk metadata integrity**
    - **Validates: Requirements 1.3, 1.4**
    - Use `hypothesis` to generate random filenames, page counts (1–50), and text lengths (0–5000 chars)
    - Assert: every chunk's `metadata.source` equals the input filename
    - Assert: every chunk's `metadata.page` is within [1, page_count]
    - Assert: chunk indices within a page are unique and non-negative
    - `# Feature: multimodal-rag, Property 1: Chunk metadata integrity`
    - _Requirements: 1.3, 1.4_

  - [x] 2.5 Write property test for `Chunker` — chunk count non-negativity
    - **Property 7: Chunk count non-negativity**
    - **Validates: Requirements 1.3**
    - Use `hypothesis` to generate random page content lists
    - Assert: total chunk count >= 0 for any input
    - Assert: chunk count equals the number of chunks in the returned list
    - `# Feature: multimodal-rag, Property 7: Chunk count non-negativity`
    - _Requirements: 1.3_

- [x] 3. Embedding service
  - [x] 3.1 Implement `EmbeddingService` in `src/rag/embedding/embedding_service.py`
    - Wrap `sentence_transformers.SentenceTransformer` with the configured model name
    - Implement `encode(texts: list[str]) -> list[list[float]]` using batched encoding
    - Raise `EmbeddingError` with chunk index and detail on failure
    - Define `EmbeddingBackend` Protocol in `src/rag/embedding/protocols.py`
    - _Requirements: 2.1, 2.2, 2.4, 2.5_

  - [x] 3.2 Write property test for `EmbeddingService` — embedding order preservation and batch consistency
    - **Property 2: Embedding order preservation**
    - **Validates: Requirements 2.1, 2.3, 2.5**
    - Use `hypothesis` to generate random lists of 1–20 non-empty strings
    - Assert: `len(encode(texts)) == len(texts)`
    - Assert: encoding the list as a batch produces the same vectors as encoding each string individually (within floating-point tolerance)
    - `# Feature: multimodal-rag, Property 2: Embedding order preservation`
    - _Requirements: 2.1, 2.3, 2.5_

  - [x] 3.3 Write unit tests for `EmbeddingService`
    - Test embedding dimensionality (384 for `all-MiniLM-L6-v2`)
    - Test that empty input list returns empty output list
    - _Requirements: 2.1, 2.2_

- [x] 4. Vector store
  - [x] 4.1 Implement `VectorStore` in `src/rag/storage/vector_store.py`
    - Use `chromadb.PersistentClient` with the configured persist directory
    - Implement `add(collection_name, chunks, embeddings)` using `upsert` with deterministic IDs (`{source}_{page}_{chunk_index}`)
    - Implement `search(collection_name, query_embedding, top_k, filter_source=None) -> list[SearchResult]`
    - Implement `list_collections() -> list[str]`
    - Implement `delete_collection(collection_name) -> None`
    - Auto-create collection if it does not exist on `add` or `search`
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7_

  - [x] 4.2 Write property test for `VectorStore` — upsert idempotence
    - **Property 3: Upsert idempotence**
    - **Validates: Requirements 3.5**
    - Use `hypothesis` to generate random lists of chunks with random source filenames
    - Ingest the same document twice into a temporary ChromaDB collection
    - Assert: the number of stored items after the second ingestion equals the number after the first
    - `# Feature: multimodal-rag, Property 3: Upsert idempotence`
    - _Requirements: 3.5_

  - [x] 4.3 Write unit tests for `VectorStore`
    - Test `list_collections` returns newly created collection
    - Test `delete_collection` removes the collection from the list
    - Test `search` on a non-existent collection returns empty list without raising
    - _Requirements: 3.3, 3.6, 3.7_

- [x] 5. Checkpoint — core components
  - Ensure all tests pass, ask the user if questions arise.

- [x] 6. Ingestion pipeline
  - [x] 6.1 Implement `IngestionPipeline` in `src/rag/ingestion/pipeline.py`
    - Accept `PDFParser`, `Chunker`, `EmbeddingBackend`, `VectorStore` via constructor
    - Implement `ingest(file_path, collection_name) -> IngestionResult`
    - Log document name, chunk count, and processing duration at INFO level
    - Return `IngestionResult` with status, chunk count, and collection name
    - Return informative status (not exception) when chunk count is zero
    - _Requirements: 1.1, 1.6, 1.7, 11.4_

  - [x] 6.2 Write unit tests for `IngestionPipeline`
    - Use mock `PDFParser`, `Chunker`, `EmbeddingBackend`, `VectorStore`
    - Verify `EmbeddingBackend.encode` is called with the chunk texts
    - Verify `VectorStore.add` is called with the correct collection name
    - Verify zero-chunk case returns status message without raising
    - _Requirements: 1.1, 1.6, 1.7_

- [x] 7. Retriever
  - [x] 7.1 Implement `Retriever` in `src/rag/retrieval/retriever.py`
    - Accept `EmbeddingBackend` and `VectorStore` via constructor
    - Implement `retrieve(query, collection_name, top_k, filter_source=None) -> RetrievalResult`
    - Log query string, result count, and retrieval duration at INFO level
    - Return `RetrievalResult` with empty chunks list and status message when collection is empty or missing
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 11.5_

  - [x] 7.2 Write property test for `Retriever` — retrieval result ordering
    - **Property 4: Retrieval result ordering**
    - **Validates: Requirements 4.2, 4.4**
    - Use `hypothesis` to generate random query strings and k values (1–10)
    - Seed a temporary collection with random chunks before each test
    - Assert: `len(results) <= top_k`
    - Assert: results are ordered by descending score (`results[i].score >= results[i+1].score`)
    - Assert: each result contains non-empty chunk text and valid metadata
    - `# Feature: multimodal-rag, Property 4: Retrieval result ordering`
    - _Requirements: 4.2, 4.3, 4.4_

  - [x] 7.3 Write property test for `Retriever` — empty retrieval graceful handling
    - **Property 6: Empty retrieval graceful handling**
    - **Validates: Requirements 4.5**
    - Use `hypothesis` to generate random query strings
    - Call `retrieve` against an empty collection and a non-existent collection name
    - Assert: returns `RetrievalResult` with empty `chunks` list
    - Assert: `status` is a non-empty string
    - Assert: no exception is raised
    - `# Feature: multimodal-rag, Property 6: Empty retrieval graceful handling`
    - _Requirements: 4.5_

- [x] 8. Response generator
  - [x] 8.1 Define `LLMBackend` Protocol in `src/rag/generation/protocols.py`
    - `complete(prompt: str) -> str`

  - [x] 8.2 Implement `OpenAICompatibleBackend` in `src/rag/generation/llm_backends.py`
    - Use `httpx` to POST to the configured `llm_api_base` with the model name and prompt
    - Raise `GenerationError` on HTTP error or timeout
    - _Requirements: 5.6_

  - [x] 8.3 Implement `ResponseGenerator` in `src/rag/generation/response_generator.py`
    - Accept `LLMBackend` via constructor
    - Build a prompt that includes the query and numbered retrieved chunks with source labels
    - Parse the LLM response to extract citations (source, page, chunk_index) from the search results
    - Return `GeneratedResponse(answer, citations)`
    - Return empty citations list with "no relevant information found" answer when chunks list is empty
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

  - [x] 8.4 Write property test for `ResponseGenerator` — citation completeness
    - **Property 5: Citation completeness**
    - **Validates: Requirements 5.3, 5.4**
    - Use `hypothesis` to generate random non-empty lists of `SearchResult` objects
    - Use a mock `LLMBackend` that returns a fixed answer string
    - Assert: every `Citation` in the response references a source, page, and chunk_index that exists in the input `SearchResult` list
    - Assert: `answer` is a non-empty string
    - `# Feature: multimodal-rag, Property 5: Citation completeness`
    - _Requirements: 5.3, 5.4_

  - [x] 8.5 Write unit tests for `ResponseGenerator`
    - Test empty chunks input returns empty citations and "no relevant information" answer
    - Test prompt construction includes query text and chunk texts
    - _Requirements: 5.1, 5.5_

- [ ] 9. Checkpoint — pipeline components
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 10. FastAPI backend
  - [ ] 10.1 Create `src/rag/api/dependencies.py`
    - Define FastAPI `Depends` providers for `Settings`, `EmbeddingService`, `VectorStore`, `IngestionPipeline`, `Retriever`, `ResponseGenerator`
    - Use module-level singletons (initialized once at startup) to avoid re-loading the embedding model on every request
    - _Requirements: 10.5_

  - [ ] 10.2 Implement `POST /ingest` endpoint in `src/rag/api/routes/ingest.py`
    - Accept `UploadFile` (PDF) and optional `collection_name` form field
    - Save to a temp file, call `IngestionPipeline.ingest()`, return `IngestResponse`
    - _Requirements: 6.1, 1.1_

  - [ ] 10.3 Implement `POST /query` endpoint in `src/rag/api/routes/query.py`
    - Accept `QueryRequest` JSON body
    - Call `Retriever.retrieve()` then `ResponseGenerator.generate()`
    - Return `GeneratedResponse`
    - _Requirements: 6.2_

  - [ ] 10.4 Implement `GET /collections`, `DELETE /collections/{name}`, `GET /health` endpoints in `src/rag/api/routes/collections.py` and `src/rag/api/routes/health.py`
    - _Requirements: 6.3, 6.4, 6.5_

  - [ ] 10.5 Add global exception handlers in `src/rag/api/main.py`
    - Map `IngestionError` → HTTP 422, `EmbeddingError`/`RetrievalError`/`GenerationError` → HTTP 500
    - Return structured `{"error": ..., "detail": ...}` body
    - Add request logging middleware (method, path, status code)
    - _Requirements: 6.6, 6.7, 11.2, 11.3_

  - [ ] 10.6 Write API integration tests using FastAPI `TestClient`
    - Test `POST /ingest` with a real PDF fixture: verify 200 response and chunk_count > 0
    - Test `POST /query` with a seeded collection: verify answer and citations in response
    - Test `GET /collections`: verify list response
    - Test `DELETE /collections/{name}`: verify collection is removed
    - Test `GET /health`: verify `{"status": "ok"}` response
    - Test `POST /ingest` with invalid file: verify HTTP 422 response
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8_

- [ ] 11. MCP tool layer
  - [ ] 11.1 Implement `MCPIngestTool` and `MCPRetrieveTool` in `src/rag/tools/mcp_tools.py`
    - Each tool wraps the corresponding pipeline component
    - `run()` returns a JSON-serializable dict with `status`, `result`, and `tool_name` keys
    - Define `input_schema` as a dict matching the MCP tool-call schema format
    - _Requirements: 8.1, 8.2, 8.3, 8.4_

  - [ ] 11.2 Implement `MCPToolRegistry` in `src/rag/tools/registry.py`
    - Register `MCPIngestTool` and `MCPRetrieveTool`
    - Implement `list_tools() -> list[dict]` returning name, description, and input_schema for each tool

  - [ ] 11.3 Add `GET /tools` endpoint in `src/rag/api/routes/tools.py`
    - Return `MCPToolRegistry.list_tools()` response
    - _Requirements: 8.5_

  - [ ] 11.4 Write unit tests for MCP tools
    - Test `MCPIngestTool.run()` with a mock pipeline: verify JSON response structure
    - Test `MCPRetrieveTool.run()` with a mock retriever: verify JSON response structure
    - Test `GET /tools` endpoint: verify all tools appear with correct schema fields
    - _Requirements: 8.1, 8.2, 8.3, 8.5_

- [ ] 12. Streamlit frontend
  - [ ] 12.1 Implement `src/rag/frontend/app.py`
    - Sidebar: fetch and display collections from `GET /collections`; add delete button per collection
    - Main area: PDF file uploader → `POST /ingest` → display ingestion result (chunk count, collection name)
    - Query text input → `POST /query` → display answer text
    - Render citations as a `st.dataframe` or `st.table` below the answer
    - Display `st.error()` for any failed API call with the error detail
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7_

- [ ] 13. Integration test: end-to-end ingest and query flow
  - [ ] 13.1 Write `tests/integration/test_ingest_query_flow.py`
    - Start the FastAPI app with `TestClient`
    - Upload a real PDF fixture via `POST /ingest`
    - Submit a query known to be answerable from the PDF via `POST /query`
    - Assert: answer is non-empty string
    - Assert: at least one citation references the uploaded PDF filename
    - _Requirements: 1.1, 4.1, 5.3, 6.1, 6.2_

  - [ ] 13.2 Write `tests/integration/test_mcp_tools.py`
    - Invoke `MCPIngestTool.run()` and `MCPRetrieveTool.run()` directly with a real PDF fixture
    - Assert: responses are JSON-serializable dicts with required keys
    - _Requirements: 8.1, 8.2, 8.3_

- [ ] 14. Final checkpoint — all tests pass
  - Ensure all tests pass, ask the user if questions arise.

---

## Notes

- Tasks marked with `*` are optional and can be skipped for a faster MVP delivery
- Each task references specific requirements for traceability
- Checkpoints at steps 5, 9, and 14 ensure incremental validation
- Property tests use `hypothesis` with minimum 100 examples per property
- The embedding model is loaded once at startup via the dependency injection layer to avoid repeated initialization overhead
- The `conftest.py` should provide: a temporary ChromaDB directory fixture, a mock `LLMBackend` fixture, and a small PDF fixture file
- Future extension hooks (image ingestion, voice query, evaluation pipeline, causal inference) are architectural placeholders — no implementation tasks are included for them in this MVP
