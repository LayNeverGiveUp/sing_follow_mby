"""Download, verify and safely install optional query WAV assets."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
from urllib.request import Request, urlopen
import zipfile


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "assets" / "query_assets.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Install optional one-click and next-line playback WAV clips.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--archive", type=Path, help="Use a local ZIP instead of downloading the Release asset")
    parser.add_argument("--force", action="store_true", help="Replace a different installed asset version")
    args = parser.parse_args()
    target = install_assets(args.manifest, ROOT, args.archive, args.force)
    print(f"Assets ready: {target}")


def install_assets(
    manifest_path: Path,
    project_root: Path,
    archive_override: Path | None = None,
    force: bool = False,
) -> Path:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = {
        "catalog_id", "version", "asset_name", "url", "sha256", "archive_size_bytes",
        "uncompressed_size_bytes", "file_count", "target",
    }
    missing = required - set(manifest)
    if missing:
        raise ValueError(f"Asset manifest is missing: {', '.join(sorted(missing))}")

    target_relative = _safe_relative_path(str(manifest["target"]))
    expected_prefix = target_relative.as_posix().rstrip("/") + "/"
    target = project_root.joinpath(*target_relative.parts)
    marker = target / ".asset-version.json"
    if marker.is_file():
        installed = json.loads(marker.read_text(encoding="utf-8"))
        installed_wavs = sum(1 for _ in target.rglob("*.wav"))
        if (
            installed.get("version") == manifest["version"]
            and installed.get("sha256") == manifest["sha256"]
            and installed_wavs == int(manifest["file_count"])
        ):
            return target
    if target.is_dir() and any(target.iterdir()) and not force:
        raise FileExistsError(f"{target} already contains assets; rerun with --force to replace them")

    cache_dir = project_root / "data" / ".asset_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archive_override or cache_dir / str(manifest["asset_name"])
    if archive_override is None and not _matches_manifest(archive_path, manifest):
        _download(str(manifest["url"]), archive_path)
    _verify_archive(archive_path, manifest, expected_prefix)

    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="query-assets-", dir=target.parent) as temporary:
        staging_root = Path(temporary)
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(staging_root)
        staged_target = staging_root.joinpath(*target_relative.parts)
        marker_payload = {
            "catalog_id": manifest["catalog_id"],
            "version": manifest["version"],
            "sha256": manifest["sha256"],
            "file_count": manifest["file_count"],
        }
        (staged_target / ".asset-version.json").write_text(
            json.dumps(marker_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        backup = target.with_name(target.name + ".previous")
        if backup.exists():
            shutil.rmtree(backup)
        if target.exists():
            target.replace(backup)
        try:
            staged_target.replace(target)
        except Exception:
            if backup.exists() and not target.exists():
                backup.replace(target)
            raise
        if backup.exists():
            shutil.rmtree(backup)
    return target


def _download(url: str, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = Request(url, headers={"User-Agent": "hum-song-followup-asset-installer"})
    try:
        with urlopen(request, timeout=60) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _verify_archive(path: Path, manifest: dict, expected_prefix: str) -> None:
    if not _matches_manifest(path, manifest):
        raise ValueError(f"Asset checksum or size mismatch: {path}")
    total_size = 0
    file_count = 0
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            normalized = _safe_relative_path(info.filename).as_posix()
            if not normalized.startswith(expected_prefix):
                raise ValueError(f"Unexpected path in asset archive: {info.filename}")
            mode = info.external_attr >> 16
            if mode and stat.S_ISLNK(mode):
                raise ValueError(f"Symbolic links are not allowed in asset archive: {info.filename}")
            if not info.is_dir():
                total_size += info.file_size
                file_count += 1
    if file_count != int(manifest["file_count"]):
        raise ValueError(f"Asset file count mismatch: expected {manifest['file_count']}, got {file_count}")
    if total_size != int(manifest["uncompressed_size_bytes"]):
        raise ValueError(f"Asset uncompressed size mismatch: expected {manifest['uncompressed_size_bytes']}, got {total_size}")


def _matches_manifest(path: Path, manifest: dict) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == int(manifest["archive_size_bytes"])
        and _sha256(path) == str(manifest["sha256"])
    )


def _safe_relative_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"Unsafe asset path: {value}")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
