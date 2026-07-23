from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def separate_vocals(audio_path: Path, output_dir: Path, mode: str, model: str) -> Path:
    if mode == "none":
        return audio_path
    output_dir.mkdir(parents=True, exist_ok=True)
    if mode == "audio-separator":
        return _with_audio_separator(audio_path, output_dir, model)
    if mode == "demucs":
        return _with_demucs(audio_path, output_dir)
    raise ValueError(f"Unknown separation mode: {mode}")


def _with_audio_separator(audio_path: Path, output_dir: Path, model: str) -> Path:
    try:
        from audio_separator.separator import Separator
    except ImportError as exc:
        raise RuntimeError(
            "audio-separator is not installed. Run: pip install -r requirements-separation.txt"
        ) from exc
    _ensure_ffmpeg_on_path()
    # audio-separator writes every stem it produces. Keep those temporary and
    # copy only vocals into the project's durable catalog directory.
    with tempfile.TemporaryDirectory(prefix="song-followup-separation-") as temporary:
        temp_dir = Path(temporary)
        separator = Separator(output_dir=str(temp_dir))
        separator.load_model(model_filename=model)
        outputs = separator.separate(str(audio_path))
        candidates = [Path(item) if Path(item).is_absolute() else temp_dir / item for item in outputs]
        candidates.extend(temp_dir.glob("*vocal*"))
        vocal = next((item for item in candidates if item.exists() and "vocal" in item.name.lower()), None)
        if vocal is None:
            raise RuntimeError(f"audio-separator did not produce a vocals stem for {audio_path}")
        target = output_dir / f"{audio_path.stem}.wav"
        shutil.copyfile(vocal, target)
        return target


def _ensure_ffmpeg_on_path() -> None:
    if shutil.which("ffmpeg"):
        return
    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise RuntimeError("FFmpeg is required. Install ffmpeg or imageio-ffmpeg.") from exc
    binary = Path(imageio_ffmpeg.get_ffmpeg_exe())
    if not binary.exists():
        raise RuntimeError("imageio-ffmpeg did not provide an executable ffmpeg binary")
    bin_dir = Path(tempfile.mkdtemp(prefix="song-followup-ffmpeg-"))
    link = bin_dir / "ffmpeg"
    link.symlink_to(binary)
    import os

    os.environ["PATH"] = f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"


def _with_demucs(audio_path: Path, output_dir: Path) -> Path:
    if shutil.which("demucs") is None:
        # The package's module entry point is also supported in virtual environments.
        command = [sys.executable, "-m", "demucs"]
    else:
        command = ["demucs"]
    work_dir = output_dir / ".demucs-work"
    try:
        subprocess.run(
            [*command, "--two-stems=vocals", "--out", str(work_dir), str(audio_path)],
            check=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("Demucs vocal separation failed. Install demucs or use pre-separated vocal WAV files.") from exc
    matches = list(work_dir.glob(f"*/{audio_path.stem}/vocals.wav"))
    if not matches:
        raise RuntimeError(f"Demucs did not produce vocals.wav for {audio_path}")
    target = output_dir / f"{audio_path.stem}_vocals.wav"
    shutil.copyfile(matches[0], target)
    return target
