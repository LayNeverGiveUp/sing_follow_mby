"""Build a deterministic, versioned ZIP for GitHub Release query assets."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import zipfile


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPOSITORY = "LayNeverGiveUp/sing_follow_mby"


def main() -> None:
    parser = argparse.ArgumentParser(description="Package optional per-line WAV clips for a GitHub Release.")
    parser.add_argument("--catalog-id", default="mao_buyi_v1")
    parser.add_argument("--version", default="v1")
    parser.add_argument("--repository", default=DEFAULT_REPOSITORY)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "dist")
    args = parser.parse_args()

    source_dir = args.source_dir or ROOT / "data" / "queries" / args.catalog_id
    archive_name = f"{args.catalog_id}-queries-{args.version}.zip"
    release_tag = f"assets-{args.catalog_id}-{args.version}"
    archive_path = args.output_dir / archive_name
    wav_paths = sorted(path for path in source_dir.rglob("*.wav") if path.is_file())
    if not wav_paths:
        raise SystemExit(f"No WAV clips found in {source_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for source_path in wav_paths:
            relative = source_path.relative_to(source_dir)
            archive_name_in_zip = PurePosixPath("data", "queries", args.catalog_id, *relative.parts).as_posix()
            info = zipfile.ZipInfo(archive_name_in_zip, date_time=(2020, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            with source_path.open("rb") as source, archive.open(info, "w", force_zip64=True) as destination:
                shutil.copyfileobj(source, destination, length=1024 * 1024)

    checksum = _sha256(archive_path)
    uncompressed_size = sum(path.stat().st_size for path in wav_paths)
    manifest = {
        "schema_version": 1,
        "catalog_id": args.catalog_id,
        "version": args.version,
        "release_tag": release_tag,
        "asset_name": archive_path.name,
        "url": f"https://github.com/{args.repository}/releases/download/{release_tag}/{archive_path.name}",
        "sha256": checksum,
        "archive_size_bytes": archive_path.stat().st_size,
        "uncompressed_size_bytes": uncompressed_size,
        "file_count": len(wav_paths),
        "target": f"data/queries/{args.catalog_id}",
    }
    manifest_path = archive_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"Archive: {archive_path}")
    print(f"Manifest: {manifest_path}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
