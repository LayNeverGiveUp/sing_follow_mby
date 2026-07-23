"""Split a recording with deliberate silent gaps into reusable humming test cases."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import librosa
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hum_song_mvp.src.audio_io import load_mono_audio
from hum_song_mvp.src.config import load_config
from hum_song_mvp.src.recognize import recognize


def main() -> None:
    parser = argparse.ArgumentParser(description="Split deliberately separated vocal phrases on silence.")
    parser.add_argument("--audio", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--database-dir", type=Path, default=ROOT / "hum_song_mvp" / "data" / "database")
    parser.add_argument("--top-db", type=float, default=35.0, help="Silence threshold relative to the recording peak")
    parser.add_argument("--minimum-seconds", type=float, default=2.5)
    parser.add_argument("--padding-seconds", type=float, default=0.10)
    args = parser.parse_args()

    if not args.audio.exists():
        raise FileNotFoundError(f"Audio file does not exist: {args.audio}")
    samples = load_mono_audio(args.audio, 16000)
    intervals = librosa.effects.split(samples, top_db=args.top_db, frame_length=2048, hop_length=160)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config()
    padding = int(args.padding_seconds * 16000)
    rows = []
    for index, (raw_start, raw_end) in enumerate(intervals):
        start = max(0, int(raw_start) - padding)
        end = min(len(samples), int(raw_end) + padding)
        if (end - start) / 16000 < args.minimum_seconds:
            continue
        path = args.output_dir / f"segment_{len(rows):03d}.wav"
        sf.write(path, np.asarray(samples[start:end], dtype=np.float32), 16000, subtype="PCM_16")
        try:
            result = recognize(path, args.database_dir, config)
        except Exception as exc:
            result = {"accepted": False, "reason": f"{type(exc).__name__}: {exc}"}
        row = {
            "segment_id": path.stem,
            "source_start_time": round(start / 16000, 3),
            "source_end_time": round(end / 16000, 3),
            "duration_seconds": round((end - start) / 16000, 3),
            "audio_path": str(path.resolve().relative_to(ROOT)),
            "recognition": result,
            "requires_manual_lyric_label": True,
        }
        rows.append(row)
        print(f"{path.name}: {row['source_start_time']:.2f}–{row['source_end_time']:.2f}s accepted={result.get('accepted')} song={result.get('song_id')} line={result.get('current_lyric_index')}")
    manifest = args.output_dir / "segments.json"
    manifest.write_text(
        json.dumps(
            {
                "source_file": str(args.audio),
                "segmentation": {"method": "energy_silence", "top_db": args.top_db, "padding_seconds": args.padding_seconds},
                "segments": rows,
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(rows)} segments -> {manifest}")


if __name__ == "__main__":
    main()
