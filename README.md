# AttentionRetention

Real-time lecture comprehension tool. Monitors audio, detects topic boundaries, and pops up interactive questions to keep students engaged.

## How it works

1. Audio (live mic or YouTube tab) → **Deepgram** speech-to-text
2. Transcript stored in **Redis** as a sequential log
3. **Notification agent** (Claude) detects when a topic wraps up
4. **Retriever agent** pulls a relevant question from the question bank
5. Question popup appears in the UI → student answers
6. **Questioning agent** evaluates the answer and gives feedback
7. All LLM calls traced in **Arize AX**

## Stack

- **Backend** — Python, FastAPI, WebSockets
- **Speech-to-text** — Deepgram nova-3
- **Memory** — Redis (transcript log, Q&A history, vector search for question bank)
- **LLM** — Anthropic Claude (claude-sonnet-4-6)
- **Observability** — Arize AX

## Setup

```bash
# 1. Start Redis locally
docker run -d --name redis -p 6379:6379 -p 8001:8001 redis/redis-stack:latest

# 2. Install backend dependencies
cd backend
python -m venv .venv
source .venv/Scripts/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Fill in: ANTHROPIC_API_KEY, DEEPGRAM_API_KEY, ARIZE_SPACE_ID, ARIZE_API_KEY
```

## Test ingestion

```bash
cd backend
python ingest_test.py --url "https://www.youtube.com/watch?v=..."
```

Downloads the lecture audio, transcribes it with Deepgram, and stores utterances in Redis. Check `http://localhost:8001` (Redis Insight) to browse the stored transcript.

## API keys needed

| Key | Where to get it |
|-----|----------------|
| `ANTHROPIC_API_KEY` | console.anthropic.com |
| `DEEPGRAM_API_KEY` | console.deepgram.com |
| `ARIZE_SPACE_ID` / `ARIZE_API_KEY` | app.arize.com → Settings |
