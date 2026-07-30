import hashlib
import json
from pathlib import Path
import sys
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.install_query_assets import install_assets


def _manifest(tmp_path, archive, *, member, content):
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    payload = {
        "catalog_id": "test_catalog",
        "version": "v1",
        "asset_name": archive.name,
        "url": "https://example.invalid/assets.zip",
        "sha256": digest,
        "archive_size_bytes": archive.stat().st_size,
        "uncompressed_size_bytes": len(content),
        "file_count": 1,
        "target": "data/queries/test_catalog",
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_local_release_archive_is_verified_and_installed(tmp_path):
    content = b"RIFF-test-wav"
    member = "data/queries/test_catalog/song/line_000.wav"
    archive = tmp_path / "assets.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr(member, content)
    manifest = _manifest(tmp_path, archive, member=member, content=content)
    project = tmp_path / "project"

    target = install_assets(manifest, project, archive)

    assert (target / "song" / "line_000.wav").read_bytes() == content
    assert json.loads((target / ".asset-version.json").read_text())["version"] == "v1"
    assert install_assets(manifest, project, archive) == target


def test_archive_path_traversal_is_rejected(tmp_path):
    content = b"not-safe"
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("../escape.wav", content)
    manifest = _manifest(tmp_path, archive, member="../escape.wav", content=content)
    project = tmp_path / "project"

    with pytest.raises(ValueError, match="Unsafe asset path"):
        install_assets(manifest, project, archive)

    assert not (tmp_path / "escape.wav").exists()
