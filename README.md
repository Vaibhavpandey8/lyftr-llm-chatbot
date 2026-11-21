# Lyftr AI — LLM Chatbot with Key Failover & Response Caching

A full-stack resilient AI chatbot built with **React + TypeScript (Vite)**, **FastAPI**, **LiteLLM**, and **SQLite**. It maintains multi-turn conversation context per session, automatically fails over across an ordered API key pool on HTTP 429/quota/5xx errors, caches identical prompt sequences using LiteLLM's built-in cache, and reports per-request latency and token usage alongside every assistant message.

---

## 1. Live URLs

- **Frontend (Live UI):** https://lyftr-llm-chatbot-one.vercel.app
- **Backend (Live API):** https://lyftr-llm-chatbot-backend.onrender.com
- **Health Endpoint:** https://lyftr-llm-chatbot-backend.onrender.com/health
- **GitHub Repository:** https://github.com/Vaibhavpandey8/lyftr-llm-chatbot

---

## 2. Run locally

### With Docker (Recommended)

1. Copy `.env.example` to `.env` and add at least two valid LLM API keys:
   ```bash
   cp .env.example .env
   # Edit .env and set LLM_API_KEYS=key1,key2 (and optionally LLM_MODEL)
   ```
2. Start both backend (`http://localhost:8000`) and frontend (`http://localhost:3000`):
   ```bash
   make up
   ```
3. Stop and clean up containers (the named `sqlite_data` volume at `/data` persists across restarts):
   ```bash
   make down
   ```

### Without Docker

#### Backend (FastAPI + LiteLLM + SQLite)
```bash
cd backend
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell:
# .\.venv\Scripts\Activate.ps1

pip install -r requirements.txt

# Set environment variables (or create a .env at the repo root)
export LLM_API_KEYS="your_key_1,your_key_2"
export LLM_MODEL="gemini/gemini-3.8-flash"
export DATABASE_URL="sqlite:///./data/app.db"
export ALLOWED_ORIGINS="http://localhost:3000,http://127.0.0.1:3000"
export ENABLE_FAILURE_SIMULATION="true"

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

#### Frontend (React + Vite)
```bash
cd frontend
npm install
export VITE_API_URL="http://localhost:8000"
npm run dev
```
Open `http://localhost:3000`.

#### Run Backend Tests
```bash
cd backend
pytest -v
```

---

## 3. Design decisions

### Key fallback
- **Ordered deterministic pool:** `LLM_API_KEYS` is parsed into deterministic entries (`key_1`, `key_2`, ...) in declaration order. Optionally, `LLM_MODELS` can map individual keys to different LiteLLM models across providers.
- **Zero key leakage:** Raw API keys are never returned in responses, stored in SQLite, or logged. Only the safe identifier (`key_1`, `key_2`, ... or `null` on cache hit) is emitted.
- **Graceful failover & strict error contract:** Each attempt disables LiteLLM's internal single-key retry loop (`num_retries=0`) so 429 rate limits, quota exhaustion, timeouts, and 5xx errors fail over immediately to the next key in the pool. If every key fails, FastAPI returns HTTP `503` with `{"detail": "all providers unavailable"}` and never writes the failed turn to SQLite.
- **Request-scoped failure simulation:** When `ENABLE_FAILURE_SIMULATION=true`, `X-Simulate-Failure` (`key_1=429`, `key_1=429,key_2=500`, `*=429`) injects matching LiteLLM exceptions (`RateLimitError`, `InternalServerError`, `BadGatewayError`, `ServiceUnavailableError`) strictly inside that single request without poisoning subsequent requests. Active simulation rules bypass cache reads so failover behavior can always be demonstrated on demand.

### Caching
- **LiteLLM built-in cache:** The gateway initializes `litellm.caching.Cache` (in-memory `local` by default, or `disk` via `CACHE_TYPE=disk`).
- **Exact context-aware cache key:** Lookups and writes are keyed on `model + messages` (the full `[system, ...history, new_user_message]` array). Sending the exact same first message in a brand-new session with the same system prompt resolves immediately as `LLM_CACHE` with `key_id: null`, `fallback_attempts: 0`, identical reply text, stored per-request token usage, and sub-millisecond server latency. Different messages or different multi-turn contexts always miss the cache and execute `LIVE_INFERENCE`.

### SQLite session storage
- **Zero external storage services:** Uses Python's stdlib `sqlite3` in WAL mode (`PRAGMA journal_mode=WAL`) with thread-safe locking.
- **Docker persistence:** Configured via `DATABASE_URL` (e.g. `sqlite:////data/app.db`) backed by a named Docker volume (`sqlite_data:/data`).
- **Auto-initialized schema & atomic writes:** Tables (`messages` matching the exact contract plus `sessions` for persisted `system_prompt` overrides) are created automatically during FastAPI lifespan startup. User and assistant turns are committed together in a single transaction only after a successful LLM reply, ensuring failed requests (422/503) never corrupt history.

---

## 4. Verification

Assuming the backend is running at `BASE=http://localhost:8000` (or your live backend URL):

### 1. Health check
```bash
curl -s "$BASE/health" | jq
# { "status": "ok", "keys_configured": 2, "cache_enabled": true }
```

### 2. Live inference & cross-session cache hit
```bash
# First call (Session A) -> LIVE_INFERENCE, key_id="key_1"
curl -s -X POST "$BASE/chat" \
  -H "Content-Type: application/json" \
  -d '{"session_id":"s-demo-1","message":"What is the capital of France?"}' | jq

# Same first message in a new session (Session B) -> LLM_CACHE, key_id=null, faster latency_ms
curl -s -X POST "$BASE/chat" \
  -H "Content-Type: application/json" \
  -d '{"session_id":"s-demo-2","message":"What is the capital of France?"}' | jq
```

### 3. Failover with `X-Simulate-Failure`
```bash
# Force key_1 to return HTTP 429 -> succeeds via key_2 with fallback_attempts=1
curl -s -X POST "$BASE/chat" \
  -H "Content-Type: application/json" \
  -H "X-Simulate-Failure: key_1=429" \
  -d '{"session_id":"s-fail-1","message":"What is 2 + 2?"}' | jq

# Force key_1 to return HTTP 500 -> succeeds via key_2 with fallback_attempts=1
curl -s -X POST "$BASE/chat" \
  -H "Content-Type: application/json" \
  -H "X-Simulate-Failure: key_1=500" \
  -d '{"session_id":"s-fail-2","message":"What is 3 + 3?"}' | jq

# Force all keys to fail -> HTTP 503 {"detail": "all providers unavailable"} and no history written
curl -i -s -X POST "$BASE/chat" \
  -H "Content-Type: application/json" \
  -H "X-Simulate-Failure: *=429" \
  -d '{"session_id":"s-fail-all","message":"This fails cleanly"}'
```

### 4. Multi-turn context, per-message tokens & history endpoint
```bash
curl -s -X POST "$BASE/chat" \
  -H "Content-Type: application/json" \
  -d '{"session_id":"s-multi","message":"My favorite language is Python."}' | jq

curl -s -X POST "$BASE/chat" \
  -H "Content-Type: application/json" \
  -d '{"session_id":"s-multi","message":"What is my favorite language?"}' | jq

# Retrieve stored history (oldest first, without system prompt)
curl -s "$BASE/sessions/s-multi/messages" | jq
```

### 5. Validation & 404 checks
```bash
# Empty message -> HTTP 422
curl -i -s -X POST "$BASE/chat" \
  -H "Content-Type: application/json" \
  -d '{"session_id":"s-err","message":"   "}'

# Unknown session -> HTTP 404 {"detail":"session not found"}
curl -i -s "$BASE/sessions/unknown-session-id/messages"
```

---

## 5. Setup Used

- **Editor / IDE:** Antigravity IDE / VS Code
- **AI Assistant:** Google Antigravity
- **Runtime & Tooling:** Python 3.12/3.13, FastAPI, LiteLLM, SQLite, React 19, Vite 6, TypeScript 5, Docker multi-stage builds, pytest

---

## 6. Bonus features (if any)

- **Optional Session System Prompt (`system-prompt-input`):** Built into the sidebar and persisted per session in SQLite so follow-up turns inherit the override automatically.
- **Interactive Failover Test Hook Selector in UI:** A sidebar toggle (off by default) lets reviewers trigger `key_1=429`, `key_1=500`, and `*=429` directly from the browser UI to observe real-time failover badges and graceful 503 error rendering.
- **Multi-model Key Pool Support (`LLM_MODELS`):** Keys can share a single `LLM_MODEL` or span different LiteLLM providers/models (e.g., Gemini primary + Groq backup) purely via environment variables.
- **One-click Deployment Blueprints:** Included `render.yaml` and `frontend/vercel.json` for fast public deployment.
