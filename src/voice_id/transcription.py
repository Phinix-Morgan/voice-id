from __future__ import annotations

import sys
from pathlib import Path

from faster_whisper import WhisperModel


MODEL_SIZE = "small"


class VoskUnavailableError(RuntimeError):
    """Raised when the local ASR model is unavailable or fails to load.

    Kept under the original Vosk name so the app's existing
    error handler (which returns HTTP 503 for this) keeps
    working unchanged.
    """

    pass


def load_client() -> WhisperModel:

    print("Loading Whisper model...")

    try:
        model = WhisperModel(
            MODEL_SIZE,
            device="cpu",
            compute_type="int8",
        )

    except Exception as exc:
        raise VoskUnavailableError(
            "Local ASR model failed to load. "
            "Please try the transcription again."
        ) from exc

    print("Whisper model loaded successfully.")

    return model


def upload_audio(
    model: WhisperModel,
    audio_path: Path,
) -> Path:

    if not audio_path.exists():
        raise FileNotFoundError(
            f"Audio file not found: {audio_path}"
        )

    print("Audio ready.")

    return audio_path


def transcribe_audio(
    model: WhisperModel,
    audio_file: Path,
) -> dict:

    segments_iter, info = model.transcribe(
        str(audio_file),
        vad_filter=True,
        beam_size=5,
    )

    segments = []

    for segment in segments_iter:

        text = segment.text.strip()

        if not text:
            continue

        segments.append(
            {
                "start": float(
                    segment.start
                ),
                "end": float(
                    segment.end
                ),
                "text": text,
            }
        )

    return {
        "language": info.language,
        "segments": segments,
    }


def validate_transcript(
    result: dict,
) -> dict:

    if not isinstance(result, dict):
        raise ValueError(
            "Transcript must be a JSON object."
        )

    if "segments" not in result:
        raise ValueError(
            "Transcript response does not contain "
            "'segments'."
        )

    if not isinstance(
        result["segments"],
        list,
    ):
        raise ValueError(
            "'segments' must be a list."
        )

    validated_segments = []

    for segment in result["segments"]:

        if not isinstance(
            segment,
            dict,
        ):
            continue

        if not all(
            key in segment
            for key in (
                "start",
                "end",
                "text",
            )
        ):
            continue

        try:
            start = float(
                segment["start"]
            )
            end = float(
                segment["end"]
            )

        except (
            TypeError,
            ValueError,
        ):
            continue

        text = str(
            segment["text"]
        ).strip()

        if not text:
            continue

        if start < 0:
            continue

        if end < start:
            continue

        validated_segments.append(
            {
                "start": start,
                "end": end,
                "text": text,
            }
        )

    validated_segments.sort(
        key=lambda segment:
        segment["start"]
    )

    return {
        "language": result.get(
            "language"
        ),
        "segments": validated_segments,
    }


def print_transcript(
    result: dict,
) -> None:

    print()
    print("=" * 70)
    print("TRANSCRIPT")
    print("=" * 70)

    language = result.get(
        "language"
    )

    if language:
        print(
            f"Language: {language}"
        )

    print()

    segments = result[
        "segments"
    ]

    if not segments:
        print(
            "No speech detected."
        )
        return

    for index, segment in enumerate(
        segments,
        start=1,
    ):
        print(
            f"[{index:02d}] "
            f"{segment['start']:07.2f}s → "
            f"{segment['end']:07.2f}s"
        )

        print(
            f"     {segment['text']}"
        )

        print()


def main() -> None:

    if len(sys.argv) != 2:
        print(
            "Usage:\n"
            '  uv run python -m voice_id.transcription '
            '"path/to/audio"'
        )

        raise SystemExit(1)

    audio_path = Path(
        sys.argv[1]
    )

    print("=" * 70)
    print("VOICE TRANSCRIPTION")
    print("=" * 70)

    print(
        f"Model: whisper-{MODEL_SIZE}"
    )

    print(
        f"Audio: {audio_path}"
    )

    try:
        client = load_client()

        audio_file = upload_audio(
            client,
            audio_path,
        )

        print()
        print("Transcribing...")

        result = transcribe_audio(
            client,
            audio_file,
        )

        result = validate_transcript(
            result
        )

        print_transcript(
            result
        )

        print("=" * 70)
        print("DONE")
        print("=" * 70)

    except VoskUnavailableError as exc:
        print()
        print("=" * 70)
        print("TRANSCRIPTION UNAVAILABLE")
        print("=" * 70)
        print()
        print(str(exc))
        print()

        raise SystemExit(2)

    except (
        FileNotFoundError,
        RuntimeError,
        ValueError,
    ) as exc:
        print()
        print("=" * 70)
        print("TRANSCRIPTION FAILED")
        print("=" * 70)
        print()
        print(str(exc))
        print()

        raise SystemExit(1)


if __name__ == "__main__":
    main()