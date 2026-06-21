import asyncio
import json
import os
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "backend"))

import instrumentation
instrumentation.setup()

import anthropic
import httpx
import redis.asyncio as aioredis

from config import settings
from services.redis_service import (
    make_text_client,
    get_recent_transcript,
    transcript_length,
    upsert_question,
)

MIN_UTTERANCES = 5       # don't consider asking until session has at least this many
MIN_NEW_UTTERANCES = 6   # minimum new utterances since last question before re-checking
CONTEXT_WINDOW = 15      # number of recent utterances to pass to Claude

# Per-session throttle: maps session_id -> transcript length when last question was generated
_last_question_at: dict[str, int] = {}

_anthropic_client: anthropic.AsyncAnthropic | None = None


def _get_client() -> anthropic.AsyncAnthropic:
    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
    return _anthropic_client


async def is_good_time_to_ask(recent: list[str], client: anthropic.AsyncAnthropic) -> bool:
    """Fast Haiku check: has the lecturer just finished a thought or reached a natural pause?"""
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
                f"Recent transcript:\n{transcript_text}"
            ),
        }],
    )
    return resp.content[0].text.strip().upper().startswith("YES")


async def generate_question(recent: list[str], client: anthropic.AsyncAnthropic) -> dict:
    """Use Claude Opus to generate an MCQ from the recent transcript."""
    transcript_text = "\n".join(f"[{i + 1}] {u}" for i, u in enumerate(recent))
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
                        "options": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 4,
                            "maxItems": 4,
                        },
                        "correct_answer": {
                            "type": "string",
                            "enum": ["A", "B", "C", "D"],
                        },
                        "topic": {"type": "string"},
                        "difficulty": {
                            "type": "string",
                            "enum": ["easy", "medium", "hard"],
                        },
                    },
                    "required": ["question", "options", "correct_answer", "topic", "difficulty"],
                    "additionalProperties": False,
                },
            }
        },
        messages=[{
            "role": "user",
            "content": (
                "You are an expert educator. Based on the following lecture transcript, "
                "generate one multiple-choice comprehension question with exactly 4 options (A–D).\n\n"
                f"Recent transcript:\n{transcript_text}\n\n"
                "Return JSON with fields: question (string), options (array of 4 answer strings), "
                "correct_answer ('A'|'B'|'C'|'D'), topic (string), difficulty ('easy'|'medium'|'hard')."
            ),
        }],
    )
    raw = next(b.text for b in resp.content if b.type == "text")
    return json.loads(raw)


API_BASE = os.environ.get("QUESTION_API_URL", "http://10.43.110.207:3000").rstrip("/")


async def send_question(question_data: dict) -> str | None:
    """POST MCQ to /api/send and return the user's selected answer (A/B/C/D)."""
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


def is_answer_correct(answer: str, correct: str) -> bool:
    return answer.strip().upper() == correct.upper()


async def generate_explanation(
    question_data: dict, answer: str, recent: list[str], client: anthropic.AsyncAnthropic
) -> str:
    transcript_text = "\n".join(f"[{i + 1}] {u}" for i, u in enumerate(recent))
    options_text = "\n".join(f"{chr(65+i)}. {opt}" for i, opt in enumerate(question_data["options"]))
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
            f"Lecture transcript for context:\n{transcript_text}\n\n"
            "Write a brief, friendly explanation of why the correct answer is right."
        )}],
    )
    return resp.content[0].text.strip()


async def send_feedback(question_id: str, session_id: str, explanation: str) -> None:
    async with httpx.AsyncClient() as http:
        try:
            r = await http.post(f"{API_BASE}/api/feedback", json={
                "question_id": question_id,
                "session_id": session_id,
                "explanation": explanation,
            }, timeout=10.0)
            r.raise_for_status()
            print(f"[api] feedback delivered → {r.status_code}")
        except Exception as exc:
            print(f"[api] send_feedback failed: {exc}")


async def on_transcript_push(session_id: str, redis: aioredis.Redis) -> None:
    """Handle a new utterance pushed onto transcript:{session_id}."""
    try:
        length = await transcript_length(session_id, redis)

        if length < MIN_UTTERANCES:
            print(f"[{session_id}] skip: only {length}/{MIN_UTTERANCES} utterances")
            return

        last_at = _last_question_at.get(session_id, 0)
        if length - last_at < MIN_NEW_UTTERANCES:
            print(f"[{session_id}] skip: only {length - last_at}/{MIN_NEW_UTTERANCES} new utterances since last question")
            return

        recent = await get_recent_transcript(session_id, redis, n=CONTEXT_WINDOW)
        if not recent:
            return

        client = _get_client()

        # Step 1: fast, cheap classification
        if not await is_good_time_to_ask(recent, client):
            return

        # Step 2: generate the comprehension question
        question_data = await generate_question(recent, client)

        # Commit the throttle position only after a question is actually generated
        _last_question_at[session_id] = length

        qid = str(uuid.uuid4())
        payload = {
            "question_id": qid,
            "session_id": session_id,
            "transcript_position": length,
            **question_data,
        }

        await upsert_question(session_id, qid, payload, redis)
        print(f"\n[{session_id}] pos={length} topic={question_data['topic']} difficulty={question_data['difficulty']}")
        print(f"Q: {question_data['question']}\n")

        answer = await send_question(question_data)
        if not answer:
            return

        if is_answer_correct(answer, question_data["correct_answer"]):
            print(f"[{session_id}] correct answer ({answer})")
        else:
            print(f"[{session_id}] incorrect answer ({answer}, correct: {question_data['correct_answer']}) — generating explanation…")
            explanation = await generate_explanation(question_data, answer, recent, client)
            await send_feedback(qid, session_id, explanation)

    except Exception as exc:
        print(f"[{session_id}] error in on_transcript_push: {exc}")


async def main() -> None:
    redis = make_text_client()
    try:
        pubsub = redis.pubsub()
        await pubsub.subscribe("transcript-events")

        print("Listening for transcript utterances…")
        async for msg in pubsub.listen():
            if msg["type"] != "message":
                continue

            session_id = msg["data"]
            print(f"[{session_id}] utterance received")
            asyncio.create_task(on_transcript_push(session_id, redis))

    finally:
        await pubsub.aclose()
        await redis.aclose()


if __name__ == "__main__":
    asyncio.run(main())
