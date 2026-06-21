import redis.asyncio as aioredis

from backend.config import settings

# --- Key patterns ---
# transcript:{session_id}                 List   — ordered transcript utterances
# summary:{session_id}                    String — rolling compressed summary
# session:{session_id}:questions          Hash   — all Q&A for session, keyed by question_id
# qbank:{uuid}                            Hash   — baseline question bank entry + embedding
# idx:questions                           Index  — FT vector index over qbank:* hashes

EMBEDDING_DIM = 384  # BAAI/bge-small-en-v1.5


def make_text_client() -> aioredis.Redis:
    return aioredis.from_url(settings.redis_url, decode_responses=True)


def make_binary_client() -> aioredis.Redis:
    """Needed for vector search — FT.SEARCH returns binary embedding fields."""
    return aioredis.from_url(settings.redis_url, decode_responses=False)


async def ensure_vector_index(redis: aioredis.Redis) -> None:
    """Create FT vector index over qbank:* hashes if it doesn't exist.
    Requires Redis Stack (Search module). Skipped gracefully if unavailable."""
    try:
        await redis.execute_command("FT.INFO", "idx:questions")
        return  # index already exists
    except Exception as e:
        if "unknown command" in str(e).lower():
            print("WARNING: Redis Search module not available — vector index skipped.")
            print("         Upgrade to Redis Stack to enable question bank search.")
            return
        # Index doesn't exist yet — create it
    try:
        await redis.execute_command(
            "FT.CREATE", "idx:questions",
            "ON", "HASH",
            "PREFIX", "1", "qbank:",
            "SCHEMA",
            "topic", "TEXT",
            "text", "TEXT",
            "answer", "TEXT",
            "difficulty", "NUMERIC",
            "embedding", "VECTOR", "HNSW", "6",
                "TYPE", "FLOAT32",
                "DIM", str(EMBEDDING_DIM),
                "DISTANCE_METRIC", "COSINE",
        )
        print("Created vector index idx:questions")
    except Exception as e:
        print(f"WARNING: Could not create vector index: {e}")


# --- Transcript helpers ---

async def append_transcript(session_id: str, text: str, redis: aioredis.Redis) -> None:
    await redis.rpush(f"transcript:{session_id}", text)
    await redis.publish("transcript-events", session_id)

async def get_recent_transcript(
    session_id: str, redis: aioredis.Redis, n: int = 20
) -> list[str]:
    return await redis.lrange(f"transcript:{session_id}", -n, -1)


async def get_full_transcript(session_id: str, redis: aioredis.Redis) -> list[str]:
    return await redis.lrange(f"transcript:{session_id}", 0, -1)


async def transcript_length(session_id: str, redis: aioredis.Redis) -> int:
    return await redis.llen(f"transcript:{session_id}")


# --- Q&A helpers ---

async def upsert_question(
    session_id: str,
    question_id: str,
    data: dict,
    redis: aioredis.Redis,
) -> None:
    """Write or update a question entry in session:{session_id}:questions."""
    import json
    existing_raw = await redis.hget(f"session:{session_id}:questions", question_id)
    existing = json.loads(existing_raw) if existing_raw else {}
    existing.update(data)
    await redis.hset(
        f"session:{session_id}:questions",
        question_id,
        json.dumps(existing),
    )


async def get_question(
    session_id: str,
    question_id: str,
    redis: aioredis.Redis,
) -> dict | None:
    import json
    raw = await redis.hget(f"session:{session_id}:questions", question_id)
    return json.loads(raw) if raw else None


async def get_all_questions(session_id: str, redis: aioredis.Redis) -> list[dict]:
    import json
    raw = await redis.hgetall(f"session:{session_id}:questions")
    return [json.loads(v) for v in raw.values()]


# --- Vector search helpers (require binary client: decode_responses=False) ---

async def store_question_embedding(
    question_id: str,
    question_text: str,
    embedding: bytes,
    redis: aioredis.Redis,
) -> None:
    await redis.hset(f"qbank:{question_id}", mapping={"text": question_text, "embedding": embedding})


async def find_similar_questions(
    query_vec: bytes,
    n: int,
    redis: aioredis.Redis,
) -> list[str]:
    try:
        results = await redis.execute_command(
            "FT.SEARCH", "idx:questions",
            f"*=>[KNN {n} @embedding $vec AS score]",
            "PARAMS", "2", "vec", query_vec,
            "SORTBY", "score",
            "RETURN", "1", "text",
            "DIALECT", "2",
        )
        texts = []
        for i in range(1, len(results), 2):
            fields = results[i + 1] if i + 1 < len(results) else []
            fd = dict(zip(fields[::2], fields[1::2]))
            text = fd.get(b"text")
            if text:
                texts.append(text.decode())
        return texts
    except Exception:
        return []


# --- Session memory helpers ---

async def get_session_memory(session_id: str, redis: aioredis.Redis) -> dict[str, dict[str, int]]:
    raw = await redis.hgetall(f"session:{session_id}:memory")
    memory: dict[str, dict[str, int]] = {}
    for field, val in raw.items():
        topic, metric = field.rsplit(":", 1)
        memory.setdefault(topic, {})[metric] = int(val)
    return memory


async def update_session_memory(
    session_id: str, topic: str, correct: bool, redis: aioredis.Redis
) -> None:
    key = f"session:{session_id}:memory"
    pipe = redis.pipeline()
    pipe.hincrby(key, f"{topic}:asked", 1)
    if correct:
        pipe.hincrby(key, f"{topic}:correct", 1)
    await pipe.execute()
