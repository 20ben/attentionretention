"""
Live system audio -> Deepgram streaming -> Redis

Captures whatever is playing through your speakers (WASAPI loopback),
transcribes it in real time with Deepgram, and stores final utterances
in Redis under transcript:{session_id}.

Usage:
    python live_test.py                        # capture default speaker
    python live_test.py --session-id my-test   # specify session id
    python live_test.py --list-devices         # show all speakers
    python live_test.py --device "Realtek"     # match speaker by name
"""

import instrumentation

instrumentation.setup()

import argparse
import queue
import threading
import time
import uuid

import redis as redis_sync
import soundcard as sc
from deepgram import DeepgramClient
from deepgram.core.events import EventType
from deepgram.listen import ListenV1Results

from config import settings

CAPTURE_RATE = 48000   # native WASAPI stereo rate
DEEPGRAM_RATE = 16000  # Deepgram receives 16 kHz
DOWNSAMPLE = CAPTURE_RATE // DEEPGRAM_RATE  # 3x decimation
BLOCK_SIZE = 4800      # 100 ms at 48 kHz


def find_speaker(name_hint: str | None):
    if name_hint:
        matches = [s for s in sc.all_speakers() if name_hint.lower() in s.name.lower()]
        if not matches:
            raise RuntimeError(f"No speaker matching '{name_hint}'. Run --list-devices to see options.")
        return matches[0]
    return sc.default_speaker()


def run(session_id: str, speaker) -> None:
    r = redis_sync.from_url(settings.redis_url, decode_responses=True)
    count = 0

    loopback = sc.get_microphone(speaker.id, include_loopback=True)
    print(f"Capturing: {speaker.name}")
    print(f"Session:   {session_id}")
    print("Initializing audio capture...")

    # Audio queue — recording thread fills it, main thread drains it to Deepgram.
    # Keeps the WASAPI buffer from overflowing during Deepgram connection setup.
    audio_queue: queue.Queue[bytes] = queue.Queue(maxsize=200)
    stop_rec = threading.Event()

    def recording_thread() -> None:
        with loopback.recorder(samplerate=CAPTURE_RATE, channels=2) as mic:
            while not stop_rec.is_set():
                try:
                    data = mic.record(numframes=BLOCK_SIZE)
                except Exception:
                    break
                mono = data.mean(axis=1)
                decimated = mono[::DOWNSAMPLE]
                pcm = (decimated * 32767).astype("int16")
                try:
                    audio_queue.put_nowait(pcm.tobytes())
                except queue.Full:
                    audio_queue.get_nowait()  # drop oldest chunk to make room
                    audio_queue.put_nowait(pcm.tobytes())

    rec_thread = threading.Thread(target=recording_thread, daemon=True)
    rec_thread.start()

    # Let soundcard warm up and fill a few chunks before connecting to Deepgram
    time.sleep(0.5)

    print("Play audio on your computer. Press Ctrl+C to stop.\n")

    client = DeepgramClient(api_key=settings.deepgram_api_key)

    with client.listen.v1.connect(
        model="nova-3",
        encoding="linear16",
        sample_rate=DEEPGRAM_RATE,
        smart_format=True,
        punctuate=True,
    ) as conn:

        def on_message(event) -> None:
            nonlocal count
            if not isinstance(event, ListenV1Results):
                return
            if not event.is_final:
                return
            try:
                sentence = event.channel.alternatives[0].transcript
            except (AttributeError, IndexError):
                return
            if sentence.strip():
                r.rpush(f"transcript:{session_id}", sentence)
                count += 1
                print(f"[{count}] {sentence}")

        conn.on(EventType.MESSAGE, on_message)
        conn.start_listening()

        try:
            while True:
                try:
                    pcm_bytes = audio_queue.get(timeout=1)
                except queue.Empty:
                    continue
                conn.send_media(pcm_bytes)
        except KeyboardInterrupt:
            pass
        finally:
            stop_rec.set()
            try:
                conn.send_close_stream()
            except Exception:
                pass

    r.close()
    instrumentation.shutdown()
    print(f"\nStopped. {count} utterances stored under session:{session_id}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture system audio (loopback) and transcribe with Deepgram"
    )
    parser.add_argument("--session-id", default=str(uuid.uuid4()), help="Redis session ID")
    parser.add_argument("--device", default=None, help="Speaker name substring to match")
    parser.add_argument("--list-devices", action="store_true", help="Print all speakers and exit")
    args = parser.parse_args()

    if args.list_devices:
        for s in sc.all_speakers():
            print(f"  {s.name}")
        return

    speaker = find_speaker(args.device)
    run(args.session_id, speaker)


if __name__ == "__main__":
    main()
