"""
Test script: YouTube URL -> yt-dlp -> Deepgram -> Redis transcript list.

Usage:
    python ingest_test.py --url "https://www.youtube.com/watch?v=..."
    python ingest_test.py --url "..." --session-id my-test-session
"""

import argparse
import asyncio
import os
import sys
import tempfile
import uuid
from pathlib import Path

import yt_dlp
from deepgram import DeepgramClient

sys.path.insert(0, str(Path(__file__).parent))

from config import settings
from services.redis_service import (
    make_text_client,
    ensure_vector_index,
    append_transcript,
    transcript_length,
    get_recent_transcript,
)


def download_audio(url: str, output_dir: str) -> str:
    """Download best available audio from YouTube, return path to file."""
    output_template = os.path.join(output_dir, "audio.%(ext)s")
    ydl_opts = {
        "format": "bestaudio[ext=m4a]/bestaudio[ext=webm]/bestaudio/best",
        "outtmpl": output_template,
        "quiet": False,
        "no_warnings": False,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)
    return filename


def transcribe_file(audio_path: str) -> list[str]:
    """
    Send audio to Deepgram pre-recorded API.
    Returns list of transcript sentences split from the full response.
    """
    client = DeepgramClient(api_key=settings.deepgram_api_key)

    print("      Sending to Deepgram (this may take a moment)...")
    with open(audio_path, "rb") as audio_file:
        response = client.listen.v1.media.transcribe_file(
            request=audio_file.read(),
            model="nova-3",
            smart_format=True,
            punctuate=True,
        )

    full_transcript = response.results.channels[0].alternatives[0].transcript  # type: ignore[union-attr]
    sentences = [s.strip() for s in full_transcript.split(". ") if s.strip()]
    return sentences


async def store_utterances(
    session_id: str, sentences: list[str], redis
) -> None:
    for sentence in sentences:
        await append_transcript(session_id, sentence, redis)


async def main(url: str, session_id: str) -> None:
    redis = make_text_client()

    try:
        print(f"\nSession ID : {session_id}")
        print(f"YouTube URL: {url}\n")

        await ensure_vector_index(redis)

        # Step 1: Download
        print("[1/3] Downloading audio from YouTube...")
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = download_audio(url, tmpdir)
            print(f"      File: {audio_path}")
            size_mb = os.path.getsize(audio_path) / 1_000_000
            print(f"      Size: {size_mb:.1f} MB")

            # Step 2: Transcribe
            print("\n[2/3] Transcribing with Deepgram nova-3...")
            utterances = transcribe_file(audio_path)

        print(f"      Utterances received: {len(utterances)}")

        # Step 3: Store
        print("\n[3/3] Storing in Redis...")
        await store_utterances(session_id, utterances, redis)
        count = await transcript_length(session_id, redis)
        print(f"      transcript:{session_id} length: {count}")

        # Preview
        recent = await get_recent_transcript(session_id, redis, n=5)
        print("\nLast 5 utterances stored:")
        for i, t in enumerate(recent):
            print(f"  [{count - len(recent) + i}] {t}")

        print(f"\nDone. Run again with --session-id {session_id} to append more.\n")

    finally:
        await redis.aclose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest a YouTube lecture into Redis via Deepgram"
    )
    parser.add_argument("--url", required=True, help="YouTube video URL")
    parser.add_argument(
        "--session-id",
        default=str(uuid.uuid4()),
        help="Session ID to store transcript under (default: new UUID)",
    )
    args = parser.parse_args()

    asyncio.run(main(args.url, args.session_id))
