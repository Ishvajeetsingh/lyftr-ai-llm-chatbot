# Lyftr AI LLM Chatbot

A production-grade, highly resilient conversational AI chatbot built with FastAPI, LiteLLM, SQLite, and React + Vite + TypeScript. Features multi-key sequential failover, built-in LLM response caching, strict per-request token usage accounting, and simulated failure testing.

---

## 1. Live URLs

- **Frontend Application**: https://lyftr-chat-frontend.vercel.app
- **Backend API**: https://lyftr-backend-1euu.onrender.com
- **API Health Check**: https://lyftr-backend-1euu.onrender.com/health
- **GitHub Repository**: https://github.com/Ishvajeetsingh/lyftr-ai-llm-chatbot

---

## 2. Run locally: with Docker and without

### Option A: Running with Docker & Docker Compose (Recommended)

Make sure Docker and Docker Compose are installed and running.

1. **Clone the repository and prepare configuration**:
   ```bash
   cp .env.example .env
   ```
   Add your Gemini or OpenAI API keys to `LLM_API_KEYS` in `.env` (comma-separated: `key_1,key_2,...`).

2. **Start the containers using Makefile**:
   ```bash
   make up
   ```
   *Alternatively*:
   ```bash
   docker compose up -d --build
   ```

3. **Verify the services**:
   - Backend API: [http://localhost:8000/health](http://localhost:8000/health)
   - Frontend UI: [http://localhost:3000](http://localhost:3000)

4. **Stop the containers**:
   ```bash
   make down
   ```

---

### Option B: Running without Docker (Directly on Host)

#### 1. Backend Setup (Python 3.11+)

```bash
cd backend

# Create local virtual environment
python -m venv .venv

# Activate virtual environment
# On Linux/macOS:
source .venv/bin/activate
# On Windows PowerShell:
.venv\Scripts\Activate.ps1

# Install dependencies synchronously with --no-input
pip install --no-input -r requirements.txt

# Run the test suite
pytest -v

# Start the FastAPI server on port 8000
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

#### 2. Frontend Setup (Node.js 18+)

```bash
cd frontend

# Install dependencies
npm install

# Start Vite dev server on port 3000
npm run dev
```

The frontend will run on [http://localhost:3000](http://localhost:3000) and connect to the backend on `http://localhost:8000`.

---

## 3. Design decisions: key fallback, caching, SQLite session storage

### Sequential Key Fallback Architecture
- **Problem**: LLM APIs regularly experience HTTP 429 (rate limits), provider outages (5xx), timeouts, or quota exhaustion. Single API keys cause hard failures for users.
- **Solution**: The backend accepts a prioritized comma-separated list of keys (`LLM_API_KEYS=key_1,key_2,key_3`). The system iterates sequentially through the keys.
- **Failover Triggers**: 429 RateLimitError, 500/502/503/504 Server Errors, timeouts, quota exhaustion, and header-simulated provider errors.
- **Circuit Termination**: If all keys in the pool fail, the system short-circuits and raises a clean HTTP `503 Service Unavailable` with `{"detail": "all providers unavailable"}`.
- **Fail-Safe Persistence**: Failed requests (`422` or `503`) strictly do NOT write anything to SQLite, preventing ghost or corrupted conversation states.

### LLM Response Caching
- **Implementation**: Utilizes LiteLLM's local cache engine (`litellm.cache = Cache(type="local")`) augmented by deterministic cache key generation.
- **Key Determinism**: The cache key is computed from `(model, full_messages_list)`.
- **Behavior**: An identical first prompt with an identical system prompt across separate sessions hits the cache instantly.
- **Metadata Transparency**: A cached response returns `resolution_source: "LLM_CACHE"`, `key_id: null`, `fallback_attempts: 0`, identical stored tokens, identical reply text, and sub-millisecond to low millisecond `latency_ms`.

### SQLite Session & Conversation History
- **Database Engine**: Standard, zero-dependency SQLite with automated parent directory creation and connection handling.
- **Schema**:
  - `sessions`: Stores `session_id` (PRIMARY KEY), optional `system_prompt`, and `created_at`.
  - `messages`: Stores `id`, `session_id`, `role`, `content`, `metadata` (JSON serialized), and `created_at`.
- **Context Injection**: Each new turn loads previous conversation messages for that `session_id` in chronological order (`ORDER BY id ASC`), binds the session's active system prompt, appends the user's latest query, and sends the full message array to LiteLLM.
- **Clean Output**: `GET /sessions/{session_id}/messages` returns conversation history excluding the system prompt.

---

## 4. Verification: how to see failover, cache hits and per-message token usage

### 1. Verify Health Endpoint
```bash
curl -X GET "https://lyftr-backend-1euu.onrender.com/health"
```
**Expected Response (200 OK)**:
```json
{
  "status": "ok",
  "keys_configured": 2,
  "cache_enabled": true
}
```

### 2. Live Inference (Cache Miss)
```bash
curl -X POST "https://lyftr-backend-1euu.onrender.com/chat" \
  -H "Content-Type: application/json" \
  -d '{"session_id": "test-session-1", "message": "What is the capital of France?", "system_prompt": "You are a concise assistant."}'
```
**Expected Metadata**:
- `"resolution_source": "LIVE_INFERENCE"`
- `"key_id": "key_1"`
- `"fallback_attempts": 0`

### 3. Cache Hit (Identical Prompt in New Session)
```bash
curl -X POST "https://lyftr-backend-1euu.onrender.com/chat" \
  -H "Content-Type: application/json" \
  -d '{"session_id": "test-session-2", "message": "What is the capital of France?", "system_prompt": "You are a concise assistant."}'
```
**Expected Metadata**:
- `"resolution_source": "LLM_CACHE"`
- `"key_id": null`
- `"fallback_attempts": 0`
- Identical reply and identical `input_tokens` / `output_tokens`

### 4. Simulating Key 1 Failure (Failover to Key 2)
```bash
curl -X POST "https://lyftr-backend-1euu.onrender.com/chat" \
  -H "Content-Type: application/json" \
  -H "X-Simulate-Failure: key_1=429" \
  -d '{"session_id": "test-session-3", "message": "Explain gravity in five words"}'
```
**Expected Metadata**:
- `"resolution_source": "LIVE_INFERENCE"`
- `"key_id": "key_2"`
- `"fallback_attempts": 1`

### 5. Simulating Total Provider Outage (HTTP 503)
```bash
curl -X POST "https://lyftr-backend-1euu.onrender.com/chat" \
  -H "Content-Type: application/json" \
  -H "X-Simulate-Failure: *=429" \
  -d '{"session_id": "test-session-4", "message": "This request must fail"}'
```
**Expected Status & Response**: `503 Service Unavailable`
```json
{
  "detail": "all providers unavailable"
}
```
Verify no database record was created:
```bash
curl -X GET "https://lyftr-backend-1euu.onrender.com/sessions/test-session-4/messages"
# Returns: 404 {"detail": "session not found"}
```

### 6. Per-Message Token Usage
Observe the `input_tokens` and `output_tokens` in the assistant message metadata. On multi-turn conversations, notice that `output_tokens` reflects solely the tokens generated for that specific turn, confirming that token counting is per-request rather than cumulative.

---

## 5. Setup Used: your editor and AI tools

- **Editor / Development Environment**: Visual Studio Code with Google Antigravity Agentic IDE Extension.
- **AI Tooling & Coding Assistants**: Google DeepMind Gemini 3.8 Flash (High) via Antigravity Agentic Framework.
- **Testing & Verification**: Pytest 8 with HTTPX asynchronous test client and automated mock response fixtures.
- **Containerization**: Multi-stage Docker builds with Alpine Linux and Python 3.11 slim images.

---

## 6. Bonus features (if any): what you added beyond the requirements and why

1. **Interactive Failure Simulator Selector in the Frontend**:
   - Added an intuitive UI selector in the sidebar allowing developers and reviewers to simulate `key_1=429`, `key_1=500`, and `*=429` on the fly without needing terminal curl commands.
2. **Visual Fallback Attempts & Key Identifier Badges**:
   - Added color-coded badges to each assistant bubble displaying the exact key used (`key_1`, `key_2`) and the number of fallback attempts incurred.
3. **Real-time Backend Health & Cache Indicator**:
   - The UI sidebar probes the `/health` endpoint on load to verify backend connectivity, number of configured keys, and cache operational status.
4. **Resilient Offline Fallback Token Estimator**:
   - Implemented an intelligent token fallback calculator in the event a custom or proxy LLM provider omits the `usage` block, ensuring token counts never return `null` or break UI assertions.
5. **Zero-Trace Failed Session Protection**:
   - Built a strict transactional guard ensuring neither the session registry nor the messages table stores partial traces when requests abort with `422` validation or `503` provider errors.
