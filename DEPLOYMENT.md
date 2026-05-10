    # Deployment Guide

## Prerequisites

1. **Google OAuth Client ID** — [console.cloud.google.com](https://console.cloud.google.com)
   - Create a project → APIs & Services → Credentials → OAuth 2.0 Client ID
   - Application type: Web application
   - Authorized JavaScript origins: `https://your-app.vercel.app`
   - Authorized redirect URIs: `https://your-app.vercel.app`
   - Copy the Client ID

2. **Cloud PostgreSQL** — [neon.tech](https://neon.tech) (free tier available)
   - Create a project → copy the connection string
   - Format: `postgresql://user:password@host/dbname?sslmode=require`

3. **Resend API key** — [resend.com](https://resend.com)
   - Add and verify your sending domain
   - Create an API key

---

## Backend — Railway

1. Push code to GitHub
2. New project → Deploy from GitHub repo
3. Set environment variables:

```
DATABASE_URL=postgresql://...
JWT_SECRET_KEY=<openssl rand -hex 32>
GOOGLE_CLIENT_ID=<your-client-id>.apps.googleusercontent.com
RESEND_API_KEY=re_...
EMAIL_FROM=noreply@yourdomain.com
CHROMA_PERSIST_DIR=/app/chroma_db
LLM_PROVIDER=groq
LLM_API_BASE=https://api.groq.com/openai/v1
LLM_MODEL=llama-3.3-70b-versatile
LLM_API_KEY=gsk_your_groq_api_key
ALLOWED_ORIGINS=https://your-app.vercel.app
```

4. Railway auto-detects `Dockerfile.backend` via `railway.json`
5. Note your Railway URL: `https://rag-backend-xxx.railway.app`

---

## Backend — Render

1. New Web Service → Connect GitHub repo
2. Render reads `render.yaml` automatically
3. Set the `sync: false` env vars in the Render dashboard
4. Note your Render URL: `https://rag-backend.onrender.com`

---

## Frontend — Vercel

1. Import GitHub repo into Vercel
2. Framework preset: **Vite**
3. Root directory: `rag-frontend`
4. Set environment variables:

```
VITE_GOOGLE_CLIENT_ID=<your-client-id>.apps.googleusercontent.com
VITE_API_BASE_URL=https://your-backend.railway.app
```

5. Deploy — Vercel reads `vercel.json` for SPA routing

---

## Local development (no Docker)

```bash
# Backend
cp .env.example .env
# Fill in GOOGLE_CLIENT_ID, RESEND_API_KEY, etc.
.venv\Scripts\uvicorn rag.api.main:app --reload

# Frontend
cd rag-frontend
cp .env.example .env
# Fill in VITE_GOOGLE_CLIENT_ID
npm run dev
```

## Local development (Docker)

```bash
# Fill in GOOGLE_CLIENT_ID and RESEND_API_KEY in docker-compose.yml
docker compose up --build
```

---

## Production CORS

Update `ALLOWED_ORIGINS` in the backend env to include your Vercel URL:

```
ALLOWED_ORIGINS=https://your-app.vercel.app
```

The backend reads this via `Settings.allowed_origins` and passes it to FastAPI's CORSMiddleware.
