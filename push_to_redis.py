"""Manually push transcript utterances to Redis to exercise trigger.py end-to-end.

Usage:
    python push_to_redis.py                    # default session, 8 utterances
    python push_to_redis.py --session abc123   # custom session ID
    python push_to_redis.py --clear            # delete existing transcript first
    python push_to_redis.py --delay 0.5        # seconds between pushes (default 0)
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from urllib.parse import urlparse, urlunparse

sys.path.insert(0, str(Path(__file__).parent))

import redis.asyncio as aioredis

from backend.config import settings
from backend.services.redis_service import append_transcript, transcript_length


def _redis_display_url(redis_url: str) -> str:
    parsed = urlparse(redis_url)
    if parsed.password:
        netloc = parsed.netloc.replace(f":{parsed.password}@", ":***@")
        return urlunparse(parsed._replace(netloc=netloc))
    return redis_url


def _redis_connect_url(redis_url: str) -> tuple[str, dict]:
    client_kwargs: dict = {
        "decode_responses": True,
        "socket_connect_timeout": 5,
        "socket_timeout": 10,
    }
    return redis_url, client_kwargs


def _redis_fallback_urls() -> list[str]:
    urls = [settings.redis_url]
    local_url = "redis://localhost:6379"
    if settings.redis_url != local_url:
        urls.append(local_url)
    return urls


UTTERANCES = [
    "Today we're going to talk about gradient descent, which is the core optimization algorithm behind most machine learning models.",
    "The intuition is simple: imagine you're standing on a hilly landscape and you want to reach the lowest point.",
    "At each step you look at the slope around you and take a small step in the direction that goes downhill the fastest.",
    "In machine learning, that landscape is the loss surface — a high-dimensional function of all the model's parameters.",
    "The gradient tells us the direction of steepest ascent, so we move in the opposite direction to minimize the loss.",
    "The size of that step is controlled by a hyperparameter called the learning rate, often written as alpha or eta.",
    "If the learning rate is too large, we overshoot the minimum and the training loss oscillates or even diverges.",
    "If it's too small, convergence is painfully slow and we waste compute time.",
    "That's why tuning the learning rate is often the first thing practitioners do when a model isn't training well.",
    "Modern optimizers like Adam adapt the learning rate per-parameter automatically, which is why they're so popular in practice.",
]


async def main() -> None:
    session_id = "test-session-001"
    print(f"Redis URL: {_redis_display_url(settings.redis_url)}")
    print(f"Session:   {session_id}")
    last_error: Exception | None = None

    for attempt, redis_url in enumerate(_redis_fallback_urls(), start=1):
        print(f"Connecting to Redis (attempt {attempt})...")
        client = aioredis.from_url(redis_url, **_redis_connect_url(redis_url)[1])
        try:
            await client.ping()
            print("Connected.\n")

            for i, utterance in enumerate(UTTERANCES, 1):
                await append_transcript(session_id, utterance, client)
                length = await transcript_length(session_id, client)
                print(f"[{i:2d}/{len(UTTERANCES)}] len={length}  {utterance[:70]}{'…' if len(utterance) > 70 else ''}")

            print(f"\nDone. transcript:{session_id} has {await transcript_length(session_id, client)} utterances.")
            return
        except Exception as exc:
            last_error = exc
            print(f"Connection attempt {attempt} failed: {exc}")
        finally:
            await client.aclose()

    raise SystemExit(f"Unable to write to Redis after trying all configured URLs: {last_error}")


if __name__ == "__main__":
    asyncio.run(main())
