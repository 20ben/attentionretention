"""
Trigger process: subscribes to Redis pub/sub and fires comprehension questions
at natural lecture pauses.

Pipeline per utterance event:
  1. Throttle  — skip if session is too short or too few utterances since last question
  2. Timing    — Haiku classifies whether the lecturer just finished a thought
  3. Context   — parallel fetch: similar past questions (vector) + student memory
  4. Generate  — Opus produces an MCQ, avoiding repeats and targeting weak topics
  5. Deliver   — POST to the student-facing API; block until the student answers
  6. Follow-up — on incorrect answer, Opus generates an explanation and POSTs feedback

Usage:
    python trigger.py
"""
import asyncio
import json
import os
import sys
import uuid
from pathlib import Path
from typing import TypedDict

sys.path.insert(0, str(Path(__file__).parent / "backend"))

import instrumentation
instrumentation.setup()

import anthropic
import httpx
import redis.asyncio as aioredis

from config import settings
from services.redis_service import (
    make_text_client,
    make_binary_client,
    ensure_vector_index,
    get_recent_transcript,
    transcript_length,
    upsert_question,
    store_question_embedding,
    find_similar_questions,
    get_session_memory,
    update_session_memory,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MIN_UTTERANCES = 7        # minimum transcript length before any question is considered
MIN_NEW_UTTERANCES = 7    # minimum new utterances since last question before re-checking
CONTEXT_WINDOW = 15       # utterances passed to Claude as context

API_BASE = os.environ.get("QUESTION_API_URL", "http://10.43.39.220:3000").rstrip("/")

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class QuestionData(TypedDict):
    question: str
    options: list[str]
    correct_answer: str   # "A" | "B" | "C" | "D"
    topic: str
    difficulty: str       # "easy" | "medium" | "hard"

# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------

# Maps session_id -> transcript length at the time the last question was generated.
# Reserved optimistically before any await to prevent duplicate firings for adjacent utterances.
_last_question_at: dict[str, int] = {}

_anthropic_client: anthropic.AsyncAnthropic | None = None
_embed_model = None


def _get_client() -> anthropic.AsyncAnthropic:
    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
    return _anthropic_client


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return _embed_model


def _embed(text: str) -> bytes:
    return _get_embed_model().encode(text, normalize_embeddings=True).astype("float32").tobytes()

# ---------------------------------------------------------------------------
# Claude helpers
# ---------------------------------------------------------------------------

async def is_good_time_to_ask(recent: list[str], client: anthropic.AsyncAnthropic) -> bool:
    """Returns True only at natural lecture pauses; rejects mid-thought utterances."""
    transcript_text = "\n".join(f"[{i + 1}] {u}" for i, u in enumerate(recent))
    resp = await client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=8,
        messages=[{
            "role": "user",
            "content": (
                "You are deciding whether to pause a lecture and ask the student a "
                "comprehension question. Reply only YES or NO.\n\n"
                "Reply YES if the lecturer just finished explaining a concept or reached "
                "a natural pause or topic transition.\n"
                "Reply NO if the explanation is still mid-thought.\n\n"
                f"Recent video context:\n{transcript_text}"
            ),
        }],
    )
    return resp.content[0].text.strip().upper().startswith("YES")


async def generate_question(
    recent: list[str],
    client: anthropic.AsyncAnthropic,
    past_questions: list[str] | None = None,
    student_memory: dict | None = None,
) -> QuestionData:
    """Generates an MCQ; past_questions and student_memory are injected as prompt context."""
    transcript_text = "\n".join(f"[{i + 1}] {u}" for i, u in enumerate(recent))

    context_parts = []
    if past_questions:
        context_parts.append(
            "Questions already asked this session (do not repeat these concepts):\n"
            + "\n".join(f"- {q}" for q in past_questions)
        )
    if student_memory:
        weak = [
            t for t, s in student_memory.items()
            if s.get("asked", 0) > 0 and s.get("correct", 0) / s["asked"] < 0.6
        ]
        if weak:
            context_parts.append(
                f"Topics the student has struggled with (prefer these if relevant): {', '.join(weak)}"
            )
    context_block = ("\n\n" + "\n\n".join(context_parts)) if context_parts else ""

    resp = await client.messages.create(
        model="claude-opus-4-8",
        max_tokens=512,
        output_config={
            "format": {
                "type": "json_schema",
                "schema": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"},
                        "options": {"type": "array", "items": {"type": "string"}},
                        "correct_answer": {"type": "string", "enum": ["A", "B", "C", "D"]},
                        "topic": {"type": "string"},
                        "difficulty": {"type": "string", "enum": ["easy", "medium", "hard"]},
                    },
                    "required": ["question", "options", "correct_answer", "topic", "difficulty"],
                    "additionalProperties": False,
                },
            }
        },
        messages=[{
            "role": "user",
            "content": (
                "You are an expert educator. Based on the following snippet of the lecture, "
                "generate one multiple-choice comprehension question with exactly 4 options (A–D)."
                f"{context_block}\n\n"
                f"Recent video context:\n{transcript_text}\n\n"
                "Return JSON with fields: question (string), options (array of 4 answer strings), "
                "correct_answer ('A'|'B'|'C'|'D'), topic (string), difficulty ('easy'|'medium'|'hard')."
            ),
        }],
    )
    raw = next(b.text for b in resp.content if b.type == "text")
    return json.loads(raw)


async def generate_explanation(
    question_data: QuestionData, answer: str, recent: list[str], client: anthropic.AsyncAnthropic
) -> str:
    """Generates targeted remediation; called only on incorrect answers."""
    transcript_text = "\n".join(f"[{i + 1}] {u}" for i, u in enumerate(recent))
    options_text = "\n".join(f"{chr(65 + i)}. {opt}" for i, opt in enumerate(question_data["options"]))
    correct = question_data["correct_answer"]
    correct_text = question_data["options"][ord(correct) - ord("A")]
    resp = await client.messages.create(
        model="claude-opus-4-8",
        max_tokens=512,
        messages=[{"role": "user", "content": (
            f"A student answered a comprehension question incorrectly.\n\n"
            f"Question: {question_data['question']}\n"
            f"Options:\n{options_text}\n"
            f"Student selected: {answer} — Correct answer: {correct}. {correct_text}\n\n"
            f"Lecture video snippet for context:\n{transcript_text}\n\n"
            "Write a brief, friendly explanation of why the correct answer is right."
        )}],
    )
    return resp.content[0].text.strip()

# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

async def send_question(question_data: QuestionData) -> str | None:
    """POSTs the MCQ and blocks up to 5 minutes for the student's answer."""
    async with httpx.AsyncClient() as http:
        try:
            r = await http.post(f"{API_BASE}/api/send", json={
                "type": "mcq",
                "question": question_data["question"],
                "options": question_data["options"],
            }, timeout=300.0)
            r.raise_for_status()
            body = r.json()
            answer = body.get("answer") or body.get("response") or body.get("text")
            print(f"[api] question sent, got answer: {answer!r}")
            return answer
        except Exception as exc:
            print(f"[api] send_question failed: {exc}")
            return None


async def send_feedback(question_id: str, session_id: str, explanation: str) -> None:
    """POSTs the explanation for an incorrect answer to /api/feedback."""
    async with httpx.AsyncClient() as http:
        try:
            r = await http.post(f"{API_BASE}/api/feedback", json={
                "question_id": question_id,
                "session_id": session_id,
                "explanation": explanation,
            }, timeout=300.0)
            r.raise_for_status()
            print(f"[api] feedback delivered → {r.status_code}")
        except Exception as exc:
            print(f"[api] send_feedback failed: {exc}")


def is_answer_correct(answer: str, correct: str) -> bool:
    return answer.strip().upper() == correct.upper()

# ---------------------------------------------------------------------------
# Core event handler
# ---------------------------------------------------------------------------

async def on_transcript_push(session_id: str, redis: aioredis.Redis, redis_bin: aioredis.Redis) -> None:
    """Throttle-gated pipeline: timing check → context fetch → generation → delivery → follow-up."""
    last_at = _last_question_at.get(session_id, 0)
    question_committed = False
    try:
        length = await transcript_length(session_id, redis)

        if length < MIN_UTTERANCES:
            print(f"[{session_id}] skip: only {length}/{MIN_UTTERANCES} utterances")
            return

        if length - last_at < MIN_NEW_UTTERANCES:
            print(f"[{session_id}] skip: only {length - last_at}/{MIN_NEW_UTTERANCES} new since last question")
            return

        # Reserve immediately so concurrent tasks don't also pass the throttle check.
        # Released back to last_at on any path that doesn't produce a question.
        _last_question_at[session_id] = length

        recent = await get_recent_transcript(session_id, redis, n=CONTEXT_WINDOW)
        if not recent:
            _last_question_at[session_id] = last_at
            return

        client = _get_client()

        # Step 1: fast, cheap timing classification
        if not await is_good_time_to_ask(recent, client):
            _last_question_at[session_id] = last_at
            return

        # Step 2: fetch memory and similar past questions in parallel
        query_vec = _embed(" ".join(recent))
        past_qs, memory = await asyncio.gather(
            find_similar_questions(query_vec, 3, redis_bin),
            get_session_memory(session_id, redis),
        )

        # Step 3: generate the comprehension question with context
        question_data = await generate_question(recent, client, past_questions=past_qs, student_memory=memory)
        question_committed = True

        qid = str(uuid.uuid4())
        payload = {
            "question_id": qid,
            "session_id": session_id,
            "transcript_position": length,
            **question_data,
        }

        await asyncio.gather(
            upsert_question(session_id, qid, payload, redis),
            store_question_embedding(qid, question_data["question"], _embed(question_data["question"]), redis_bin),
        )
        print(f"\n[{session_id}] pos={length} topic={question_data['topic']} difficulty={question_data['difficulty']}")
        print(f"Q: {question_data['question']}\n")

        answer = await send_question(question_data)
        if not answer:
            return

        correct = is_answer_correct(answer, question_data["correct_answer"])
        await update_session_memory(session_id, question_data["topic"], correct, redis)

        if correct:
            print(f"[{session_id}] correct answer ({answer})")
        else:
            print(f"[{session_id}] incorrect ({answer}, correct: {question_data['correct_answer']}) — generating explanation…")
            explanation = await generate_explanation(question_data, answer, recent, client)
            await send_feedback(qid, session_id, explanation)

    except Exception as exc:
        if not question_committed:
            _last_question_at[session_id] = last_at
        print(f"[{session_id}] error in on_transcript_push: {exc}")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    _get_embed_model()  # warm up before any tasks start
    redis = make_text_client()
    redis_bin = make_binary_client()
    try:
        await ensure_vector_index(redis)
        pubsub = redis.pubsub()
        await pubsub.subscribe("transcript-events")

        print("Listening for transcript utterances…")
        async for msg in pubsub.listen():
            if msg["type"] != "message":
                continue

            session_id = msg["data"]
            print(f"[{session_id}] utterance received")
            asyncio.create_task(on_transcript_push(session_id, redis, redis_bin))

    finally:
        await pubsub.aclose()
        await redis.aclose()
        await redis_bin.aclose()


if __name__ == "__main__":
    asyncio.run(main())
