# Design Document: Multimodal RAG System

## Overview

The Multimodal RAG System is a modular, production-style Retrieval-Augmented Generation platform. The MVP delivers PDF ingestion, semantic vector search, and citation-aware answer generation through a FastAPI backend and Streamlit frontend. The architecture is intentionally layered so that each concern — ingestion, embedding, storage, retrieval, generation, and API — is isolated behind a well-defined interface. This makes it straightforward to swap implementations (e.g., a different embedding model, a different vector store) or add new modalities (images, audio) without touching unrelated code.

The system is also designed with MCP (Model Context Protocol) compatibility in mind: the core ingestion and retrieval capabilities are wrapped as MCP-compatible tool definitions so that external agent workflows can invoke them programmatically.

### Key Design Decisions

- **ChromaDB with local persistence** for the MVP. ChromaDB supports in-process operation with disk persistence, which eliminates infrastructure dependencies for local development while remaining replaceable with a remote vector store later.
- **sentence-transformers (`all-MiniLM-L6-v2`)** as the default embedding model. It is fast, lightweight, and produces 384-dimensional embeddings suitable for semantic search.
- **PyMuPDF (fitz)** for PDF parsing. It is robust, handles complex layouts, and exposes per-page text extraction with page number metadata.
- **Pydantic v2** for all data models and configuration. Provides runtime validation, serialization, and clear schema documentation.
- **Dependency injection via constructor parameters** rather than a DI framework. Keeps the code simple and testable without adding framework overhead.
- **LLM backend abstraction** with a default OpenAI-compatible HTTP client. The `Response_Generator` accepts any callable that matches the `LLMBackend` protocol, making it trivial to swap in a local model (Ollama, LM Studio) or a different provider.

---

## Architecture

The system is organized into a layered architecture with clear data flow:

```
┌─────────────────────────────────────────────────────────────┐
│                        Frontend (Streamlit)                  │
└────────────────────────────┬────────────────────────────────┘
                             │ HTTP
┌────────────────────────────▼────────────────────────────────┐
│                      API Server (FastAPI)                    │
│  /ingest  /query  /collections  /health  /tools             │
└──┬──────────────┬──────────────────────────────┬────────────┘
   │              │                              │
   ▼              ▼                              ▼
Ingestion     Retriever                   MCP Tool Layer
Pipeline      ┌──────────┐               (thin wrappers)
┌──────────┐  │Embedding │
│PDF Parser│  │Service   │
│Chunker   │  └────┬─────┘
│Embedder  │       │
└────┬─────┘       │
     │             ▼
     └──────► Vector Store (ChromaDB)
                   │
                   ▼
             Response Generator
             (LLM Backend)
```

### Data Flow: Ingestion

1. User uploads PDF via Frontend → `POST /ingest`
2. `API_Server` saves the file to a temp path and calls `IngestionPipeline.ingest(file_path, collection_name)`
3. `IngestionPipeline` uses `PDFParser` to extract per-page text
4. `Chunker` splits page text into overlapping `Chunk` objects with metadata
5. `EmbeddingService.encode(chunks)` produces embedding vectors
6. `VectorStore.add(collection_name, chunks, embeddings)` persists everything
7. Response: `{status, chunk_count, collection_name}`

### Data Flow: Query

1. User submits query via Frontend → `POST /query`
2. `API_Server` calls `Retriever.retrieve(query, collection_name, top_k)`
3. `Retriever` calls `EmbeddingService.encode([query])` to get query vector
4. `VectorStore.search(collection_name, query_vector, top_k)` returns ranked chunks
5. `ResponseGenerator.generate(query, chunks)` builds prompt and calls LLM
6. Response: `{answer, citations: [{source, page, chunk_index, score}]}`

---

## Components and Interfaces

### 1. PDFParser

Responsible for extracting text from PDF files on a per-page basis.

```python
class PDFParser:
    def parse(self, file_path: str) -> list[PageContent]:
        """Extract text from each page. Returns list of (page_number, text) pairs."""
```

- Uses `PyMuPDF` (`fitz`) internally.
- Returns `PageContent(page_number: int, text: str)` for each page.
- Raises `IngestionError` if the file cannot be opened or parsed.

### 2. Chunker

Splits page text into overlapping chunks.

```python
class Chunker:
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64): ...
    def chunk(self, pages: list[PageContent], source_filename: str) -> list[Chunk]: ...
```

- Splits on sentence/word boundaries where possible (character-level fallback).
- Attaches `DocumentMetadata(source=filename, page=page_number, chunk_index=i)` to each chunk.
- Configurable `chunk_size` (characters) and `chunk_overlap`.

### 3. EmbeddingService

Converts text to dense vectors.

```python
class EmbeddingService:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"): ...
    def encode(self, texts: list[str]) -> list[list[float]]: ...
```

- Wraps `sentence_transformers.SentenceTransformer`.
- Supports batched encoding.
- Raises `EmbeddingError` on failure.

**Interface (Protocol) for extensibility:**

```python
class EmbeddingBackend(Protocol):
    def encode(self, texts: list[str]) -> list[list[float]]: ...
```

### 4. VectorStore

Persists and retrieves embeddings.

```python
class VectorStore:
    def __init__(self, persist_directory: str, embedding_service: EmbeddingBackend): ...
    def add(self, collection_name: str, chunks: list[Chunk], embeddings: list[list[float]]) -> None: ...
    def search(self, collection_name: str, query_embedding: list[float], top_k: int, filter_source: str | None = None) -> list[SearchResult]: ...
    def list_collections(self) -> list[str]: ...
    def delete_collection(self, collection_name: str) -> None: ...
```

- Uses `chromadb.PersistentClient`.
- `add()` uses `upsert` semantics keyed on a deterministic ID (`{source}_{page}_{chunk_index}`) to handle re-ingestion.
- `search()` returns `SearchResult(chunk: Chunk, score: float)`.

### 5. IngestionPipeline

Orchestrates the full ingestion flow.

```python
class IngestionPipeline:
    def __init__(self, parser: PDFParser, chunker: Chunker, embedding_service: EmbeddingBackend, vector_store: VectorStore): ...
    def ingest(self, file_path: str, collection_name: str) -> IngestionResult: ...
```

- Returns `IngestionResult(status: str, chunk_count: int, collection_name: str)`.
- Logs document name, chunk count, and duration.

### 6. Retriever

Executes semantic search.

```python
class Retriever:
    def __init__(self, embedding_service: EmbeddingBackend, vector_store: VectorStore): ...
    def retrieve(self, query: str, collection_name: str, top_k: int = 5, filter_source: str | None = None) -> RetrievalResult: ...
```

- Returns `RetrievalResult(chunks: list[SearchResult], status: str)`.
- Logs query, result count, and duration.

### 7. ResponseGenerator

Builds prompts and calls the LLM.

```python
class ResponseGenerator:
    def __init__(self, llm_backend: LLMBackend): ...
    def generate(self, query: str, search_results: list[SearchResult]) -> GeneratedResponse: ...
```

- Returns `GeneratedResponse(answer: str, citations: list[Citation])`.
- Prompt template injects retrieved chunks with source labels so the LLM can cite them.
- `LLMBackend` protocol:

```python
class LLMBackend(Protocol):
    def complete(self, prompt: str) -> str: ...
```

Default implementation: `OpenAICompatibleBackend` using `httpx` to call an OpenAI-compatible endpoint.

### 8. MCP Tool Layer

Thin wrappers that expose `IngestionPipeline` and `Retriever` as MCP-compatible tools.

```python
class MCPIngestTool:
    name = "rag_ingest"
    description = "Ingest a PDF document into the RAG system"
    input_schema = { "file_path": str, "collection_name": str }
    def run(self, file_path: str, collection_name: str) -> dict: ...

class MCPRetrieveTool:
    name = "rag_retrieve"
    description = "Retrieve relevant chunks for a query"
    input_schema = { "query": str, "collection_name": str, "top_k": int }
    def run(self, query: str, collection_name: str, top_k: int = 5) -> dict: ...
```

- Each tool's `run()` returns a JSON-serializable dict matching the MCP tool-call response schema.
- A `MCPToolRegistry` collects all tools and exposes their schemas via `GET /tools`.

### 9. API Server

FastAPI application wiring all components together.

```
POST   /ingest                  → IngestionPipeline.ingest()
POST   /query                   → Retriever.retrieve() + ResponseGenerator.generate()
GET    /collections             → VectorStore.list_collections()
DELETE /collections/{name}      → VectorStore.delete_collection()
GET    /health                  → { status: "ok", version: str }
GET    /tools                   → MCPToolRegistry.list_tools()
```

- Uses FastAPI's dependency injection (`Depends`) to provide component instances.
- Global exception handlers map `IngestionError`, `EmbeddingError`, `RetrievalError` to appropriate HTTP status codes.

### 10. Frontend

Streamlit application.

- Sidebar: collection selector (from `GET /collections`), collection deletion button.
- Main area: PDF upload widget → calls `POST /ingest` → shows result.
- Query input → calls `POST /query` → renders answer + citations table.
- Error display for failed API calls.

---

## Data Models

All models use Pydantic v2.

```python
class DocumentMetadata(BaseModel):
    source: str          # original filename
    page: int            # 1-indexed page number
    chunk_index: int     # 0-indexed position within the document

class Chunk(BaseModel):
    text: str
    metadata: DocumentMetadata

class PageContent(BaseModel):
    page_number: int
    text: str

class SearchResult(BaseModel):
    chunk: Chunk
    score: float         # cosine similarity, higher is more similar

class Citation(BaseModel):
    source: str
    page: int
    chunk_index: int
    score: float

class IngestionResult(BaseModel):
    status: str
    chunk_count: int
    collection_name: str

class RetrievalResult(BaseModel):
    chunks: list[SearchResult]
    status: str

class GeneratedResponse(BaseModel):
    answer: str
    citations: list[Citation]

class QueryRequest(BaseModel):
    query: str
    collection_name: str = "default"
    top_k: int = 5
    filter_source: str | None = None

class IngestResponse(BaseModel):
    status: str
    chunk_count: int
    collection_name: str

class HealthResponse(BaseModel):
    status: str
    version: str
```

### Configuration Model

```python
class Settings(BaseSettings):
    embedding_model: str = "all-MiniLM-L6-v2"
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k: int = 5
    chroma_persist_dir: str = "./chroma_db"
    llm_api_base: str = "http://localhost:11434/v1"  # Ollama default
    llm_model: str = "llama3"
    llm_api_key: str = "ollama"
    log_level: str = "INFO"
    app_version: str = "0.1.0"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
```

---

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system — essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Chunk metadata integrity

*For any* PDF document ingested into the system, every Chunk produced by the Ingestion_Pipeline SHALL have a DocumentMetadata whose `source` matches the input filename, `page` is within the valid page range of the document, and `chunk_index` is a non-negative integer unique within that page.

**Validates: Requirements 1.3, 1.4**

### Property 2: Embedding order preservation

*For any* list of text strings passed to `EmbeddingService.encode()`, the returned list of embeddings SHALL have the same length as the input list, and the embedding at index `i` SHALL correspond to the text at index `i`.

**Validates: Requirements 2.1, 2.3**

### Property 3: Upsert idempotence

*For any* document ingested into the Vector_Store twice with the same filename and collection name, the total number of stored chunks for that document SHALL be the same after the second ingestion as after the first — i.e., re-ingestion does not duplicate entries.

**Validates: Requirements 3.5**

### Property 4: Retrieval result ordering

*For any* query submitted to the Retriever, the returned SearchResults SHALL be ordered by descending similarity score — i.e., `results[i].score >= results[i+1].score` for all valid `i`.

**Validates: Requirements 4.2, 4.4**

### Property 5: Citation completeness

*For any* GeneratedResponse produced by the Response_Generator from a non-empty list of SearchResults, every Citation in the `citations` list SHALL reference a source, page, and chunk_index that corresponds to one of the input SearchResults.

**Validates: Requirements 5.3, 5.4**

### Property 6: Empty retrieval graceful handling

*For any* query submitted against an empty or non-existent Collection, the Retriever SHALL return a RetrievalResult with an empty `chunks` list and a non-empty `status` string, without raising an exception.

**Validates: Requirements 4.5**

### Property 7: Chunk count non-negativity

*For any* PDF document ingested, the `chunk_count` in the IngestionResult SHALL be greater than or equal to zero, and SHALL equal the number of Chunks actually stored in the Vector_Store for that document.

**Validates: Requirements 1.3, 6.1**

---

## Error Handling

### Error Taxonomy

| Error Class | Raised By | HTTP Status | Description |
|---|---|---|---|
| `IngestionError` | PDFParser, Chunker, IngestionPipeline | 422 | Document cannot be parsed or produces no chunks |
| `EmbeddingError` | EmbeddingService | 500 | Embedding model failure |
| `RetrievalError` | Retriever, VectorStore | 500 | Vector store query failure |
| `GenerationError` | ResponseGenerator | 500 | LLM backend failure |
| `ConfigurationError` | Settings | startup | Missing required config |

### API Error Response Shape

```json
{
  "error": "IngestionError",
  "detail": "Could not parse PDF: file is encrypted",
  "request_id": "abc123"
}
```

### Retry and Fallback Strategy

- **EmbeddingService**: No automatic retry in MVP. Raises `EmbeddingError` immediately on failure.
- **LLM Backend**: Single attempt in MVP. `GenerationError` is returned to the caller with the error detail.
- **VectorStore**: ChromaDB operations are synchronous and local; no retry needed for MVP.
- **Future**: An exponential backoff decorator (`@with_retry`) is defined as a placeholder hook for remote service calls.

---

## Testing Strategy

### Dual Testing Approach

The system uses both unit tests (specific examples and edge cases) and property-based tests (universal properties across generated inputs). These are complementary: unit tests catch concrete bugs in known scenarios, property tests verify general correctness across the input space.

### Property-Based Testing

**Library**: `hypothesis` (Python)

Each correctness property is implemented as a single `@given`-decorated test with a minimum of 100 examples. Tests are tagged with a comment referencing the design property.

**Tag format**: `# Feature: multimodal-rag, Property {N}: {property_text}`

Property test locations:

| Property | Test File | What is Generated |
|---|---|---|
| P1: Chunk metadata integrity | `tests/test_chunker_properties.py` | Random page counts, text lengths, filenames |
| P2: Embedding order preservation | `tests/test_embedding_properties.py` | Random lists of strings (varying length, content) |
| P3: Upsert idempotence | `tests/test_vector_store_properties.py` | Random documents ingested twice |
| P4: Retrieval result ordering | `tests/test_retriever_properties.py` | Random queries against populated collections |
| P5: Citation completeness | `tests/test_generator_properties.py` | Random search result sets |
| P6: Empty retrieval graceful handling | `tests/test_retriever_properties.py` | Random queries against empty/missing collections |
| P7: Chunk count non-negativity | `tests/test_ingestion_properties.py` | Random valid PDF-like page content |

### Unit Tests

- `tests/test_pdf_parser.py`: Parse a known PDF, verify page count and text extraction.
- `tests/test_chunker.py`: Verify chunk size bounds, overlap, metadata attachment.
- `tests/test_embedding_service.py`: Verify embedding dimensionality, batch consistency.
- `tests/test_vector_store.py`: Add, search, list, delete collections.
- `tests/test_retriever.py`: End-to-end retrieval with a seeded collection.
- `tests/test_response_generator.py`: Mock LLM backend, verify citation extraction.
- `tests/test_api.py`: FastAPI `TestClient` tests for all endpoints.

### Integration Tests

- `tests/integration/test_ingest_query_flow.py`: Upload a real PDF, query it, verify citations reference the document.
- `tests/integration/test_mcp_tools.py`: Invoke MCP tools directly, verify JSON response schema.

### Test Infrastructure

- `pytest` as the test runner.
- `pytest-asyncio` for async FastAPI tests.
- `hypothesis` for property-based tests.
- Fixtures in `conftest.py` provide a temporary ChromaDB directory and a mock LLM backend.

---

## Future Extension Hooks

The following placeholders are defined in the architecture to support future phases without requiring structural changes:

1. **Image Ingestion**: `ImageParser` implementing the same `DocumentParser` protocol as `PDFParser`. The `IngestionPipeline` accepts any `DocumentParser`, so image support is additive.
2. **Voice Query**: A `VoiceTranscriber` component that converts audio to a query string before passing it to the `Retriever`. No changes to retrieval or generation.
3. **Evaluation Pipeline**: An `EvaluationHook` protocol with a `evaluate(query, retrieved_chunks, answer)` method. The `ResponseGenerator` calls this hook if one is registered (no-op by default).
4. **Causal Inference**: A `CausalAnalyzer` component that can be injected into the query pipeline to post-process retrieved chunks before generation.
5. **Remote Vector Store**: `VectorStore` is backed by a `VectorStoreBackend` protocol. A `PineconeBackend` or `WeaviateBackend` can be swapped in by changing the `Settings`.
