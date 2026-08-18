from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


# ---------------------------------------------------------------------------
# Supported input formats
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".m4a",
    ".aac",
    ".flac",
    ".ogg",
    ".opus",
    ".webm",
}


# ---------------------------------------------------------------------------
# FFmpeg
# ---------------------------------------------------------------------------

def ensure_ffmpeg() -> None:
    """
    Make sure ffmpeg is available.

    Audio normalization relies on ffmpeg because the diarization stack
    expects an audio format that can be read reliably by soundfile.
    """

    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg is required for audio normalization but was not found. "
            "Install it with: sudo pacman -S ffmpeg"
        )


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def validate_audio_path(
    audio_path: Path,
) -> None:

    if not audio_path.exists():
        raise FileNotFoundError(
            f"Audio file not found: {audio_path}"
        )

    if not audio_path.is_file():
        raise ValueError(
            f"Audio path is not a file: {audio_path}"
        )

    extension = audio_path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(
            sorted(
                extension.lstrip(".")
                for extension in SUPPORTED_EXTENSIONS
            )
        )

        raise ValueError(
            f"Unsupported audio format: {extension or 'unknown'}\n"
            f"Supported formats: {supported}"
        )


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize_audio(
    audio_path: Path,
    output_dir: Path | None = None,
) -> Path:

    audio_path = Path(audio_path)

    validate_audio_path(
        audio_path
    )

    ensure_ffmpeg()

    if output_dir is None:
        output_dir = (
            audio_path.parent / "normalized"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / f"{audio_path.stem}.wav"
    )

    print()
    print("=" * 70)
    print("AUDIO NORMALIZATION")
    print("=" * 70)

    print(
        f"Input  : {audio_path}"
    )

    print(
        f"Output : {output_path}"
    )

    # -----------------------------------------------------------------------
    # WAV files are still normalized deliberately.
    #
    # This guarantees the same downstream format regardless of what the
    # original WAV contains.
    # -----------------------------------------------------------------------

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(audio_path),

        # Mono
        "-ac",
        "1",

        # 16 kHz
        "-ar",
        "16000",

        # PCM signed 16-bit WAV
        "-c:a",
        "pcm_s16le",

        str(output_path),
    ]

    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )

    except OSError as exc:
        raise RuntimeError(
            "Failed to execute ffmpeg."
        ) from exc

    if result.returncode != 0:

        error = result.stderr.strip()

        raise RuntimeError(
            "Audio normalization failed.\n"
            f"{error}"
        )

    if not output_path.exists():
        raise RuntimeError(
            "ffmpeg completed without creating "
            "the normalized audio file."
        )

    print(
        "Normalization complete."
    )

    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:

    import sys

    if len(sys.argv) != 2:
        print(
            "Usage:\n"
            '  uv run python -m voice_id.audio '
            '"path/to/audio"'
        )

        raise SystemExit(1)

    input_path = Path(
        sys.argv[1]
    )

    normalized_path = normalize_audio(
        input_path
    )

    print()
    print(
        f"Normalized audio: {normalized_path}"
    )


if __name__ == "__main__":
    main()
