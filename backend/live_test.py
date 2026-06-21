"""
Live system audio -> Deepgram streaming -> Redis

Captures system audio via Stereo Mix, transcribes with Deepgram, stores
final utterances in Redis under transcript:{session_id}.

Usage:
    python live_test.py                        # auto-detect loopback device
    python live_test.py --session-id my-test
    python live_test.py --list-devices
    python live_test.py --device 14
"""

import instrumentation

instrumentation.setup()

import argparse
import json
import queue
import threading
import uuid

import numpy as np
import redis as redis_sync
import sounddevice as sd
from websockets.sync.client import connect as ws_connect

from config import settings

CAPTURE_RATE = 48000
DEEPGRAM_RATE = 16000
DOWNSAMPLE = CAPTURE_RATE // DEEPGRAM_RATE  # 3x decimation
BLOCK_SIZE = 4800  # 100 ms at 48 kHz

DEEPGRAM_URL = (
    f"wss://api.deepgram.com/v1/listen"
    f"?model=nova-3"
    f"&encoding=linear16"
    f"&sample_rate={DEEPGRAM_RATE}"
    f"&smart_format=true"
    f"&punctuate=true"
)

LOOPBACK_KEYWORDS = ("stereo mix", "loopback", "what u hear", "wave out", "sum")


def find_loopback_device() -> tuple[int, str]:
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            name = dev["name"].lower()
            if any(k in name for k in LOOPBACK_KEYWORDS):
                return i, dev["name"]
    raise RuntimeError(
        "No loopback input device found.\n"
        "Run --list-devices to see inputs.\n"
        "Enable 'Stereo Mix' in Windows Sound settings if missing."
    )


def run(session_id: str, device_index: int, device_name: str) -> None:
    r = redis_sync.from_url(settings.redis_url, decode_responses=True)
    count = 0
    audio_queue: queue.Queue[bytes] = queue.Queue(maxsize=200)

    def audio_callback(indata: np.ndarray, _frames, _time, _status) -> None:
        mono = indata.mean(axis=1)
        decimated = mono[::DOWNSAMPLE]
        pcm = (decimated * 32767).astype("int16")
        try:
            audio_queue.put_nowait(pcm.tobytes())
        except queue.Full:
            audio_queue.get_nowait()
            audio_queue.put_nowait(pcm.tobytes())

    print(f"Device:  [{device_index}] {device_name}")
    print(f"Session: {session_id}")
    print("Initializing audio capture...")

    with sd.InputStream(
        device=device_index,
        samplerate=CAPTURE_RATE,
        channels=2,
        dtype="float32",
        blocksize=BLOCK_SIZE,
        callback=audio_callback,
    ):
        # Drain startup transient
        for _ in range(5):
            audio_queue.get(timeout=2)

        print("Play audio on your computer. Press Ctrl+C to stop.\n")

        headers = {"Authorization": f"Token {settings.deepgram_api_key}"}

        with ws_connect(DEEPGRAM_URL, additional_headers=headers) as ws:

            def receive_loop() -> None:
                nonlocal count
                try:
                    for raw in ws:
                        msg = json.loads(raw)
                        if msg.get("type") != "Results":
                            continue
                        if not msg.get("is_final"):
                            continue
                        try:
                            sentence = msg["channel"]["alternatives"][0]["transcript"]
                        except (KeyError, IndexError):
                            continue
                        if sentence.strip():
                            r.rpush(f"transcript:{session_id}", sentence)
                            count += 1
                            print(f"[{count}] {sentence}")
                except Exception:
                    pass

            recv_thread = threading.Thread(target=receive_loop, daemon=True)
            recv_thread.start()

            try:
                while True:
                    try:
                        pcm_bytes = audio_queue.get(timeout=1)
                    except queue.Empty:
                        continue
                    ws.send(pcm_bytes)
            except KeyboardInterrupt:
                pass
            finally:
                try:
                    ws.send(json.dumps({"type": "CloseStream"}))
                except Exception:
                    pass

    r.close()
    instrumentation.shutdown()
    print(f"\nStopped. {count} utterances stored under session:{session_id}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-id", default=str(uuid.uuid4()))
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--list-devices", action="store_true")
    args = parser.parse_args()

    if args.list_devices:
        devices = sd.query_devices()
        hostapis = sd.query_hostapis()
        for i, dev in enumerate(devices):
            if dev["max_input_channels"] > 0:
                api = hostapis[dev["hostapi"]]["name"]
                print(f"  [{i}] [{api}] {dev['name']}")
        return

    if args.device is not None:
        device_index = args.device
        device_name = sd.query_devices(device_index)["name"]
    else:
        device_index, device_name = find_loopback_device()

    run(args.session_id, device_index, device_name)


if __name__ == "__main__":
    main()
