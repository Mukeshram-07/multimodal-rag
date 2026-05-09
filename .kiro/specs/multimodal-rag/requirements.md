# Requirements Document

## Introduction

This document defines the requirements for a production-style Multimodal Retrieval-Augmented Generation (RAG) platform. The MVP focuses on PDF ingestion and semantic retrieval with citation-aware answer generation. The system is designed with a modular architecture to support future extensions including image understanding, voice query integration, evaluation pipelines, and causal inference workflows. The platform exposes a FastAPI backend, a Streamlit frontend, and is designed to be compatible with Model Context Protocol (MCP) tool integration for future agent workflows.

## Glossary

- **RAG_System**: The end-to-end retrieval-augmented generation platform described in this document.
- **Ingestion_Pipeline**: The component responsible for loading, parsing, chunking, and embedding documents.
- **Embedding_Service**: The component that converts text chunks into dense vector representations using sentence-transformers.
- **Vector_Store**: The ChromaDB-backed component that stores and retrieves vector embeddings.
- **Retriever**: The component that accepts a query, embeds it, and returns the most semantically relevant chunks from the Vector_Store.
- **Response_Generator**: The component that takes retrieved chunks and a user query and produces a citation-aware natural language answer.
- **API_Server**: The FastAPI application that exposes HTTP endpoints for ingestion, retrieval, and query answering.
- **Frontend**: The Streamlit application that provides a user interface for uploading documents and submitting queries.
- **Chunk**: A discrete segment of text extracted from a document, used as the unit of embedding and retrieval.
- **Citation**: A reference to the source document and location (page number, chunk index) from which a piece of retrieved content originates.
- **MCP_Tool**: A Model Context Protocol-compatible interface that wraps RAG_System capabilities for use by external agent workflows.
- **Collection**: A named namespace within the Vector_Store that groups embeddings for a specific set of documents.
- **Document_Metadata**: Structured information associated with a Chunk, including source filename, page number, and chunk index.

---

## Requirements

### Requirement 1: PDF Document Ingestion

**User Story:** As a user, I want to upload PDF documents to the system, so that their content becomes searchable and retrievable.

#### Acceptance Criteria

1. WHEN a user uploads a PDF file via the Frontend, THE API_Server SHALL accept the file and an optional collection name, then pass both to the Ingestion_Pipeline.
2. WHEN the Ingestion_Pipeline receives a PDF file, THE Ingestion_Pipeline SHALL extract all text content from each page of the document.
3. WHEN text is extracted from a PDF, THE Ingestion_Pipeline SHALL split the text into Chunks of configurable size (1–10,000 characters) with configurable overlap (0 to chunk_size−1 characters).
4. WHEN Chunks are created, THE Ingestion_Pipeline SHALL attach Document_Metadata to each Chunk, including source filename, 1-indexed page number, and 0-indexed chunk index.
5. IF a PDF file cannot be parsed, THEN THE Ingestion_Pipeline SHALL raise an IngestionError identifying the filename and the failure reason.
6. IF a PDF file contains no extractable text, THEN THE Ingestion_Pipeline SHALL return an IngestionResult with chunk_count of 0 and a status message indicating the document produced zero chunks.
7. WHEN Chunks are created with metadata, THE Ingestion_Pipeline SHALL pass them to the Embedding_Service for vectorization.
8. IF the uploaded file's media type is not `application/pdf`, THEN THE API_Server SHALL reject the request with an HTTP 422 error indicating the file type is not supported.
9. IF the uploaded file exceeds 50 MB, THEN THE API_Server SHALL reject the request with an HTTP 422 error indicating the file size limit was exceeded.

---

### Requirement 2: Text Embedding

**User Story:** As a developer, I want document chunks to be converted into vector embeddings, so that semantic similarity search is possible.

#### Acceptance Criteria

1. WHEN the Embedding_Service receives a list of text Chunks, THE Embedding_Service SHALL produce a dense vector representation for each Chunk using a sentence-transformers model.
2. THE Embedding_Service SHALL use a configurable model name, defaulting to `all-MiniLM-L6-v2`.
3. WHEN embeddings are produced, THE Embedding_Service SHALL return embeddings in the same order as the input Chunks, such that the embedding at index i corresponds to the Chunk at index i.
4. IF the Embedding_Service encounters an error during encoding, THEN THE Embedding_Service SHALL raise an EmbeddingError with the chunk index and error detail.
5. THE Embedding_Service SHALL encode all input Chunks in a single batched call to the underlying model.
6. WHEN the configured model is `all-MiniLM-L6-v2`, THE Embedding_Service SHALL produce embeddings of exactly 384 dimensions per Chunk.

---

### Requirement 3: Vector Storage and Indexing

**User Story:** As a developer, I want embeddings and their associated metadata to be stored in a vector database, so that they can be efficiently retrieved by semantic similarity.

#### Acceptance Criteria

1. WHEN the Ingestion_Pipeline produces embeddings and metadata, THE Vector_Store SHALL persist them in a named Collection.
2. THE Vector_Store SHALL use ChromaDB as the underlying storage engine with local disk persistence.
3. WHEN a Collection name is provided, THE Vector_Store SHALL create the Collection if it does not already exist.
4. WHEN documents are added to the Vector_Store, THE Vector_Store SHALL store the Chunk text, its embedding vector, and its Document_Metadata together as a single record.
5. IF a document with the same source filename is ingested into the same Collection a second time, THEN THE Vector_Store SHALL overwrite all existing records for that source filename using upsert semantics keyed on a deterministic ID composed of source filename, page number, and chunk index.
6. THE Vector_Store SHALL support listing all available Collections.
7. THE Vector_Store SHALL support deleting a named Collection and all its contents.

---

### Requirement 4: Semantic Retrieval

**User Story:** As a user, I want to submit a natural language query and receive the most relevant document chunks, so that I can find information across my uploaded documents.

#### Acceptance Criteria

1. WHEN a user submits a query string, THE Retriever SHALL embed the query using the Embedding_Service.
2. WHEN the query embedding is produced, THE Retriever SHALL query the Vector_Store for the top-k most similar Chunks, where k is a configurable integer between 1 and 100 inclusive, defaulting to 5.
3. WHEN results are returned, THE Retriever SHALL include the Chunk text and its Document_Metadata in each result.
4. WHEN results are returned, THE Retriever SHALL include a cosine similarity score for each result and order results by descending score such that results[i].score >= results[i+1].score for all valid i.
5. IF the Vector_Store Collection is empty or does not exist, THEN THE Retriever SHALL return a RetrievalResult with an empty chunks list and a non-empty status message, without raising an exception.
6. THE Retriever SHALL support filtering results by source document filename, returning only Chunks whose Document_Metadata source matches the specified filename.

---

### Requirement 5: Citation-Aware Answer Generation

**User Story:** As a user, I want the system to generate a natural language answer to my query that cites the source documents it used, so that I can verify the information.

#### Acceptance Criteria

1. WHEN the Retriever returns a set of relevant Chunks, THE Response_Generator SHALL construct a prompt that includes the query and the retrieved Chunk texts with their source labels.
2. WHEN generating an answer, THE Response_Generator SHALL produce a response that references the source document and page number for each piece of information used.
3. WHEN an answer is generated, THE Response_Generator SHALL return both the answer text and a structured list of Citations.
4. WHEN a Citation is returned, THE Citation SHALL include the source filename, page number, chunk index, and similarity score, all drawn from the input SearchResults.
5. IF no relevant Chunks are retrieved, THEN THE Response_Generator SHALL return a GeneratedResponse with a non-empty answer string indicating no relevant information was found and an empty Citations list.
6. THE Response_Generator SHALL use a configurable LLM backend defined by three settings: `llm_api_base` (the base URL of an OpenAI-compatible API endpoint), `llm_model` (the model name to request), and `llm_api_key` (the authentication key). The default values SHALL point to a local Ollama instance.

---

### Requirement 6: FastAPI Backend

**User Story:** As a developer, I want a well-structured REST API, so that the frontend and external tools can interact with the RAG system programmatically.

#### Acceptance Criteria

1. THE API_Server SHALL expose a `POST /ingest` endpoint that accepts a multipart form with a PDF file upload and an optional `collection_name` field (defaulting to `"default"`), and returns an IngestResponse with status, chunk_count, and collection_name.
2. THE API_Server SHALL expose a `POST /query` endpoint that accepts a QueryRequest JSON body with query, collection_name, top_k, and optional filter_source fields, and returns a GeneratedResponse with answer and citations.
3. THE API_Server SHALL expose a `GET /collections` endpoint that returns a list of available Collection names.
4. THE API_Server SHALL expose a `DELETE /collections/{name}` endpoint that deletes the specified Collection and returns a confirmation.
5. THE API_Server SHALL expose a `GET /health` endpoint that returns a HealthResponse with status `"ok"` and the application version string.
6. THE API_Server SHALL expose a `GET /tools` endpoint that returns the list of available MCP_Tools and their input schemas.
7. WHEN a request to any endpoint fails due to a validation error, THE API_Server SHALL return an HTTP 422 response with a JSON body containing `error`, `detail`, and `request_id` fields.
8. WHEN a request to any endpoint fails due to an internal error, THE API_Server SHALL return an HTTP 500 response with a JSON body containing `error`, `detail`, and `request_id` fields.
9. THE API_Server SHALL include OpenAPI documentation accessible at `/docs`.
10. THE API_Server SHALL allow cross-origin requests from any origin so that the Streamlit Frontend can call the API when running on a different port.

---

### Requirement 7: Streamlit Frontend

**User Story:** As a user, I want a simple web interface, so that I can upload documents and ask questions without using the API directly.

#### Acceptance Criteria

1. THE Frontend SHALL provide a file upload widget that accepts PDF files only.
2. WHEN a user uploads a PDF, THE Frontend SHALL call the `POST /ingest` endpoint and display the ingestion result including chunk count and collection name.
3. THE Frontend SHALL provide a text input for submitting natural language queries.
4. WHEN a user submits a query, THE Frontend SHALL call the `POST /query` endpoint using the currently selected Collection and display the generated answer.
5. WHEN an answer is displayed, THE Frontend SHALL render the Citations as a structured table below the answer showing source filename, page number, and similarity score for each Citation.
6. WHEN an API call fails, THE Frontend SHALL display the error detail from the API response using a visible error indicator.
7. THE Frontend SHALL display the list of available Collections retrieved from `GET /collections` and allow the user to select which Collection to query and to delete a Collection.
8. WHILE an API call is in progress, THE Frontend SHALL display a loading indicator and disable the triggering control until the response is received.

---

### Requirement 8: MCP-Compatible Tool Interface

**User Story:** As a developer, I want the RAG system's core capabilities exposed as MCP-compatible tools, so that external agent workflows can invoke ingestion and retrieval programmatically.

#### Acceptance Criteria

1. THE RAG_System SHALL define an MCP_Tool named `rag_ingest` for document ingestion that declares an input schema with required fields `file_path` (string) and `collection_name` (string).
2. THE RAG_System SHALL define an MCP_Tool named `rag_retrieve` for semantic retrieval that declares an input schema with required fields `query` (string) and `collection_name` (string), and optional field `top_k` (integer, default 5).
3. WHEN an MCP_Tool is invoked, THE MCP_Tool SHALL return a JSON-serializable dict containing exactly the keys `tool_name` (string), `status` (string), and `result` (dict).
4. THE MCP_Tool interfaces SHALL be implemented as thin wrappers over the existing Ingestion_Pipeline and Retriever components, containing no business logic of their own.
5. THE API_Server SHALL expose a `GET /tools` endpoint that returns a list of all registered MCP_Tools, each entry containing `name`, `description`, and `input_schema` fields.

---

### Requirement 9: Configuration Management

**User Story:** As a developer, I want all system parameters to be configurable via environment variables or a configuration file, so that the system can be adapted to different environments without code changes.

#### Acceptance Criteria

1. THE RAG_System SHALL read configuration from environment variables with sensible defaults for all fields.
2. THE RAG_System SHALL support configuration of the following named parameters: `EMBEDDING_MODEL` (string, default `all-MiniLM-L6-v2`), `CHUNK_SIZE` (integer 1–10,000, default 512), `CHUNK_OVERLAP` (integer 0 to CHUNK_SIZE−1, default 64), `TOP_K` (integer 1–100, default 5), `CHROMA_PERSIST_DIR` (string, default `./chroma_db`), `LLM_API_BASE` (string URL, default `http://localhost:11434/v1`), `LLM_MODEL` (string, default `llama3`), and `LLM_API_KEY` (string, default `ollama`).
3. WHEN a required configuration value is missing and has no default, THE RAG_System SHALL raise a ConfigurationError at startup identifying the missing key by name.
4. THE RAG_System SHALL support a `.env` file for local development configuration, with the `.env.example` file documenting all available keys and their defaults.

---

### Requirement 10: Modular Architecture and Extensibility

**User Story:** As a developer, I want the system to be organized into clearly separated modules, so that individual components can be replaced or extended without affecting the rest of the system.

#### Acceptance Criteria

1. THE RAG_System SHALL organize code into distinct modules under `src/rag/`: `ingestion`, `embedding`, `retrieval`, `generation`, `storage`, `api`, `frontend`, and `tools`.
2. WHEN a new document modality (e.g., images, audio) is added, THE Ingestion_Pipeline SHALL support it by implementing a defined ingestion interface without modifying existing ingestion code.
3. WHEN a new embedding model is introduced, THE Embedding_Service SHALL support it by implementing a defined embedding interface without modifying retrieval or storage code.
4. THE RAG_System SHALL include placeholder hooks in the architecture for future evaluation pipelines and causal inference components.
5. THE RAG_System SHALL use dependency injection via constructor parameters so that components can be swapped during testing or deployment without modifying component code.
6. THE RAG_System SHALL provide a `conftest.py` at the `tests/` root that supplies at minimum: a temporary ChromaDB directory fixture that is cleaned up after each test, a mock LLMBackend fixture that returns a fixed non-empty string, and a small valid PDF file fixture for use in integration tests.

---

### Requirement 11: Error Handling and Observability

**User Story:** As a developer, I want consistent error handling and structured logging throughout the system, so that I can diagnose issues in production.

#### Acceptance Criteria

1. THE RAG_System SHALL use structured logging with configurable log levels (DEBUG, INFO, WARNING, ERROR), defaulting to INFO.
2. WHEN an unhandled exception occurs in any component, THE RAG_System SHALL log the exception class name, message, and full stack trace at ERROR level with a contextual message identifying the component.
3. THE API_Server SHALL log each incoming request with HTTP method, path, and response status code at INFO level.
4. WHEN the Ingestion_Pipeline processes a document, THE Ingestion_Pipeline SHALL log the document name, chunk count, and processing duration in milliseconds at INFO level.
5. WHEN the Retriever executes a query, THE Retriever SHALL log the query string, number of results returned, and retrieval duration in milliseconds at INFO level.
6. THE API_Server SHALL assign a unique request_id to each incoming request and include it in all log entries and error response bodies for that request.
