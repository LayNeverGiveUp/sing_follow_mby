import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.bootstrap_runtime import DATABASE_DIR, RUNTIME_DIRECTORIES, main as bootstrap_runtime


def test_versioned_runtime_database_is_complete_and_portable():
    metadata_paths = sorted(DATABASE_DIR.glob("*.json"))

    assert len(metadata_paths) == 7
    for metadata_path in metadata_paths:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        assert metadata["lrc_lines"]
        assert not Path(metadata["audio_path"]).is_absolute()
        assert not Path(metadata["vocal_path"]).is_absolute()
        assert (DATABASE_DIR / metadata["features_file"]).is_file()


def test_bootstrap_creates_runtime_directories():
    bootstrap_runtime()

    assert all(directory.is_dir() for directory in RUNTIME_DIRECTORIES)
