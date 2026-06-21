"""
Quick soundcard loopback test — prints audio level every 100ms.
Play something on your computer and watch the levels move.

Usage:
    python soundcard_test.py
    python soundcard_test.py --seconds 10
    python soundcard_test.py --save out.wav
"""

import argparse
import struct
import wave

import numpy as np
import soundcard as sc

CAPTURE_RATE = 48000
BLOCK_SIZE = 4800  # 100 ms


def level_bar(rms: float, width: int = 40) -> str:
    filled = int(min(rms * width * 10, width))
    return f"[{'#' * filled}{' ' * (width - filled)}] {rms:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--save", default=None, help="Save to WAV file")
    args = parser.parse_args()

    speaker = sc.default_speaker()
    loopback = sc.get_microphone(speaker.id, include_loopback=True)
    print(f"Speaker:  {speaker.name}")
    print(f"Loopback: {loopback.name}")
    print(f"Capture:  {CAPTURE_RATE} Hz stereo, {args.seconds}s\n")

    blocks = int(args.seconds * CAPTURE_RATE / BLOCK_SIZE)
    all_frames = []

    with loopback.recorder(samplerate=CAPTURE_RATE, channels=2) as mic:
        for _ in range(blocks):
            data = mic.record(numframes=BLOCK_SIZE)  # (4800, 2) float64
            mono = data.mean(axis=1)
            rms = float(np.sqrt(np.mean(mono ** 2)))
            print(f"\r{level_bar(rms)}", end="", flush=True)
            all_frames.append(mono)

    print("\n\nDone.")

    if args.save:
        combined = np.concatenate(all_frames)
        pcm = (combined * 32767).astype("int16")
        with wave.open(args.save, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(CAPTURE_RATE)
            wf.writeframes(struct.pack(f"{len(pcm)}h", *pcm))
        print(f"Saved to {args.save}")


if __name__ == "__main__":
    main()
