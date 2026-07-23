"""Create browser-playable vocal clips for every timestamped LRC line."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from urllib.parse import quote

import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hum_song_mvp.src.lrc_parser import parse_lrc
from hum_song_mvp.src.build_database import filter_lyric_lines
from hum_song_mvp.src.config import load_config
from hum_song_mvp.src.recognize import recognize


def main() -> None:
    parser = argparse.ArgumentParser(description="Build static humming-MVP test clips from separated vocals.")
    parser.add_argument("--vocals-dir", type=Path, default=Path("data/source_vocals/mao_buyi_v1"))
    parser.add_argument("--lyrics-dir", type=Path, default=Path("data/source_lyrics/mao_buyi_v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/queries/mao_buyi_v1"))
    parser.add_argument("--verify", action="store_true", help="Keep only clips that the current database recognizes correctly")
    parser.add_argument("--database-dir", type=Path, default=Path("hum_song_mvp/data/database"))
    parser.add_argument("--verify-limit", type=int, default=12, help="Evenly sampled candidates to verify; 0 means all")
    args = parser.parse_args()
    created = 0
    candidates = []
    for vocal_path in sorted(args.vocals_dir.glob("*.wav")):
        lrc_path = args.lyrics_dir / f"{vocal_path.stem}.lrc"
        if not lrc_path.exists():
            raise FileNotFoundError(f"Missing LRC for {vocal_path.name}: {lrc_path}")
        info = sf.info(vocal_path)
        metadata_path = args.database_dir / f"{vocal_path.stem}.json"
        if metadata_path.exists():
            # Reuse the F0-derived vocal ends from the actual recognition DB.
            lines = json.loads(metadata_path.read_text(encoding="utf-8"))["lrc_lines"]
        else:
            lines = filter_lyric_lines(parse_lrc(lrc_path, info.duration), vocal_path.stem, info.duration)
        song_dir = args.output_dir / vocal_path.stem
        song_dir.mkdir(parents=True, exist_ok=True)
        for line in lines:
            start = int(float(line["start_time"]) * info.samplerate)
            frames = max(1, int((float(line["end_time"]) - float(line["start_time"])) * info.samplerate))
            audio, rate = sf.read(vocal_path, start=start, frames=frames, always_2d=True)
            if len(audio) == 0:
                continue
            target = song_dir / f"line_{int(line['index']):03d}.wav"
            sf.write(target, audio, rate, subtype="PCM_16")
            if line["index"] + 1 < len(lines):
                candidates.append({"song_id": vocal_path.stem, "line": line, "path": target})
            created += 1
    print(f"Built {created} vocal test clips in {args.output_dir}")
    if args.verify:
        config = load_config()
        if args.verify_limit > 0 and len(candidates) > args.verify_limit:
            positions = [round(index * (len(candidates) - 1) / (args.verify_limit - 1)) for index in range(args.verify_limit)]
            candidates = [candidates[index] for index in positions]
        verified = []
        for position, candidate in enumerate(candidates, start=1):
            try:
                result = recognize(candidate["path"], args.database_dir, config)
            except ValueError as exc:
                print(f"Verified {position}/{len(candidates)}: skipped ({exc})")
                continue
            line = candidate["line"]
            expected_index = int(line["index"])
            if (
                result.get("accepted")
                and result.get("song_id") == candidate["song_id"]
                and result.get("current_lyric_index") == expected_index
                and result.get("next_lyric_index") == expected_index + 1
            ):
                song = quote(candidate["song_id"])
                verified.append(
                    {
                        "song_id": candidate["song_id"],
                        "current_lyric_index": expected_index,
                        "current_lyric_text": line["text"],
                        "next_lyric_index": expected_index + 1,
                        "query_audio_url": f"/static/queries/mao_buyi_v1/{song}/line_{expected_index:03d}.wav",
                    }
                )
            print(f"Verified {position}/{len(candidates)}: {len(verified)} accepted")
        manifest = args.output_dir / "verified_test_queries.json"
        manifest.write_text(json.dumps({"items": verified}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {len(verified)} verified test queries -> {manifest}")


if __name__ == "__main__":
    main()
