from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from voice_id.alignment import align
from voice_id.diarization import diarize_audio
from voice_id.transcription import (
    load_client,
    transcribe_audio,
    upload_audio,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_diarization_result(
    result: Any,
) -> list[dict[str, Any]]:
    """
    Convert the diarization output into the format required by alignment.py.

    Expected format:

    [
        {
            "start": float,
            "end": float,
            "speaker": str | None,
        }
    ]
    """

    if isinstance(result, dict):
        for key in (
            "segments",
            "diarization",
            "results",
        ):
            if key in result:
                result = result[key]
                break

    if not isinstance(result, list):
        raise TypeError(
            "Unexpected diarization result. "
            "Expected a list of segments."
        )

    normalized: list[dict[str, Any]] = []

    for segment in result:
        if not isinstance(segment, dict):
            continue

        start = segment.get("start")
        end = segment.get("end")

        if start is None or end is None:
            continue

        speaker = segment.get("speaker")

        normalized.append(
            {
                "start": float(start),
                "end": float(end),
                "speaker": speaker,
            }
        )

    return normalized


def normalize_transcription_result(
    result: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Extract timestamped transcript segments from the Gemini result.
    """

    segments = result.get("segments")

    if not isinstance(segments, list):
        raise TypeError(
            "Unexpected transcription result. "
            "Expected 'segments' to be a list."
        )

    normalized: list[dict[str, Any]] = []

    for segment in segments:
        if not isinstance(segment, dict):
            continue

        start = segment.get("start")
        end = segment.get("end")
        text = segment.get("text")

        if start is None or end is None or text is None:
            continue

        text = str(text).strip()

        if not text:
            continue

        normalized.append(
            {
                "start": float(start),
                "end": float(end),
                "text": text,
            }
        )

    return normalized


def collect_speakers(
    diarization_segments: list[dict[str, Any]],
) -> list[str]:
    speakers: list[str] = []

    for segment in diarization_segments:
        speaker = segment.get("speaker")

        if speaker is None:
            continue

        if speaker not in speakers:
            speakers.append(speaker)

    return speakers


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def process_audio(
    audio_path: str | Path,
) -> dict[str, Any]:
    """
    Run the complete Voice ID pipeline.

    Audio
      ↓
    Diarization
      ↓
    Gemini transcription
      ↓
    Timestamp alignment
      ↓
    Final speaker-attributed transcript
    """

    audio_path = Path(audio_path)

    if not audio_path.exists():
        raise FileNotFoundError(
            f"Audio file not found: {audio_path}"
        )

    # -----------------------------------------------------------------------
    # 1. DIARIZATION
    # -----------------------------------------------------------------------

    print("=" * 70)
    print("VOICE ID PIPELINE")
    print("=" * 70)

    print()
    print("1. SPEAKER DIARIZATION")
    print("-" * 70)

    diarization_result = diarize_audio(
        audio_path,
    )

    diarization_segments = normalize_diarization_result(
        diarization_result,
    )

    speakers = collect_speakers(
        diarization_segments,
    )

    # -----------------------------------------------------------------------
    # 2. TRANSCRIPTION
    # -----------------------------------------------------------------------

    print()
    print("2. GEMINI TRANSCRIPTION")
    print("-" * 70)

    client = load_client()

    audio_file = upload_audio(
        client,
        audio_path,
    )

    transcription_result = transcribe_audio(
        client,
        audio_file,
    )

    transcript_segments = normalize_transcription_result(
        transcription_result,
    )

    # -----------------------------------------------------------------------
    # 3. ALIGNMENT
    # -----------------------------------------------------------------------

    print()
    print("3. SPEAKER / TRANSCRIPT ALIGNMENT")
    print("-" * 70)

    aligned_segments = align(
        diarization_segments,
        transcript_segments,
    )

    # -----------------------------------------------------------------------
    # 4. FINAL RESULT
    # -----------------------------------------------------------------------

    result = {
        "audio": {
            "file": audio_path.name,
        },
        "speakers": speakers,
        "transcription": {
            "language": transcription_result.get(
                "language"
            ),
            "segments": transcript_segments,
        },
        "diarization": diarization_segments,
        "transcript": aligned_segments,
    }

    return result


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_result(
    result: dict[str, Any],
) -> None:
    print()
    print("=" * 70)
    print("FINAL SPEAKER-ATTRIBUTED TRANSCRIPT")
    print("=" * 70)

    print()

    speakers = result["speakers"]

    print("SPEAKERS")
    print("-" * 70)

    if speakers:
        for speaker in speakers:
            print(f"  {speaker}")
    else:
        print("  None")

    print()
    print("TRANSCRIPT")
    print("-" * 70)

    transcript = result["transcript"]

    if not transcript:
        print("No transcript segments.")
        return

    for segment in transcript:
        start = segment["start"]
        end = segment["end"]
        speaker = segment["speaker"]
        text = segment["text"]

        print(
            f"[{start:07.2f}s → {end:07.2f}s]"
        )
        print(
            f"{str(speaker).upper()}"
        )
        print(text)
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) != 2:
        print(
            "Usage:\n"
            '  uv run python -m voice_id.pipeline '
            '"path/to/audio"'
        )
        raise SystemExit(1)

    audio_path = Path(sys.argv[1])

    result = process_audio(
        audio_path,
    )

    print_result(
        result,
    )

    print("=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
