# Docker Development Guide

## Prerequisites

| Tool | Version | Notes |
|---|---|---|
| Docker Desktop | ≥ 4.x | Windows / macOS |
| Docker Engine + Compose | ≥ 24.x | Linux |
| Ollama | any | Running on the **host** machine |

Ollama must be running on your host before starting the stack:

```bash
ollama serve          # starts Ollama on http://localhost:11434
ollama pull llama3:8b # pull the model if not already present
```

---

## Quick start

```bash
# 1. Clone / enter the project
cd multimodal-rag-system

# 2. First run — builds both images (takes a few minutes)
docker compose up --build

# 3. Subsequent runs — no rebuild needed
docker compose up
```

| Service | URL |
|---|---|
| React frontend | http://localhost:5173 |
| FastAPI backend | http://localhost:8000 |
| API docs (Swagger) | http://localhost:8000/docs |

---

## Development workflow

Both containers mount source directories as read-only volumes:

- **Backend**: `./src` → `/app/src` — uvicorn `--reload` watches this directory.
  Edit any Python file and the server restarts automatically.

- **Frontend**: `./rag-frontend/src`, `public/`, `index.html` → `/app/...` —
  Vite HMR picks up changes instantly in the browser.

`node_modules` is **not** mounted — it lives inside the frontend image.
If you add a new npm package, rebuild the frontend image:

```bash
docker compose build frontend
docker compose up frontend
```

---

## Stopping and cleaning up

```bash
# Stop containers (keep volumes)
docker compose down

# Stop and remove all volumes (deletes ChromaDB data and model cache)
docker compose down -v

# Remove images too
docker compose down --rmi all
```

---

## Environment variables

All backend settings are in the `environment:` block of `docker-compose.yml`.
The most important ones:

| Variable | Default (Docker) | Description |
|---|---|---|
| `LLM_API_BASE` | `http://host.docker.internal:11434` | Ollama URL |
| `LLM_MODEL` | `llama3:8b` | Model name |
| `CHROMA_PERSIST_DIR` | `/app/chroma_db` | ChromaDB data path |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformers model |
| `LOG_LEVEL` | `INFO` | Python log level |

Frontend proxy target:

| Variable | Default (Docker) | Description |
|---|---|---|
| `VITE_API_PROXY_TARGET` | `http://backend:8000` | Backend URL for Vite proxy |

---

## Troubleshooting

### "API offline" badge in the frontend

The frontend health-checks the backend via the Vite proxy (`/api/health`).
If it shows offline:

1. Check the backend is healthy: `docker compose ps`
2. Check backend logs: `docker compose logs backend`
3. Verify the backend container started: `curl http://localhost:8000/health`

### Ollama connection errors ("LLM request error")

The backend reaches Ollama via `host.docker.internal:11434`.

- **Windows / macOS**: `host.docker.internal` is resolved automatically by
  Docker Desktop. Make sure Ollama is running (`ollama serve`).
- **Linux**: The `extra_hosts: host.docker.internal:host-gateway` entry in
  `docker-compose.yml` maps the hostname to the Docker bridge gateway.
  Verify with: `docker compose exec backend ping -c1 host.docker.internal`

If Ollama is bound to `127.0.0.1` only (not `0.0.0.0`), it won't be reachable
from inside Docker on Linux. Set `OLLAMA_HOST=0.0.0.0` before starting Ollama:

```bash
OLLAMA_HOST=0.0.0.0 ollama serve
```

### Model download on first start

The sentence-transformers model (`all-MiniLM-L6-v2`) is downloaded on the
first request if not already cached. The `model_cache` named volume persists
it across container restarts. First startup may take 1–2 minutes.

### Port conflicts

If ports 8000 or 5173 are already in use, change the host-side port in
`docker-compose.yml`:

```yaml
ports:
  - "8001:8000"   # backend on host port 8001
```

---

## Non-Docker workflow (unchanged)

The Docker setup does not affect local development:

```bash
# Backend
.venv\Scripts\uvicorn rag.api.main:app --reload

# Frontend
cd rag-frontend && npm run dev
```

The local `.env` file is read by both the Python settings and Vite.
The Docker `.env.docker` file is provided as a reference but is not
loaded automatically — settings are injected via `docker-compose.yml`.

---

## Preparing for production deployment

The current setup is optimised for development (hot-reload, dev servers).
For production:

1. **Backend**: Remove `--reload` from the `CMD` in `Dockerfile.backend`.
   Add `--workers 2` (or more) to uvicorn.

2. **Frontend**: Change the frontend `CMD` to `npm run build` and serve
   `dist/` with an nginx container.

3. **Ollama**: Add an `ollama` service to `docker-compose.yml` using the
   official `ollama/ollama` image, or point `LLM_API_BASE` at a remote
   Ollama / OpenAI-compatible endpoint.

4. **Secrets**: Move `LLM_API_KEY` and other secrets to Docker secrets or
   a secrets manager rather than plain environment variables.
