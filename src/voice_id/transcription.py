from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import errors
from google.genai import types


MODEL_NAME = "gemini-3.5-flash"


class GeminiUnavailableError(RuntimeError):
    """Raised when Gemini is temporarily unavailable."""

    pass


def load_client() -> genai.Client:
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. "
            "Add it to your .env file."
        )

    return genai.Client(
        api_key=api_key
    )


def upload_audio(
    client: genai.Client,
    audio_path: Path,
):
    if not audio_path.exists():
        raise FileNotFoundError(
            f"Audio file not found: {audio_path}"
        )

    print("Uploading audio to Gemini...")

    audio_file = client.files.upload(
        file=str(audio_path),
    )

    print("Audio uploaded successfully.")

    return audio_file


def transcribe_audio(
    client: genai.Client,
    audio_file,
) -> dict:

    prompt = """
Transcribe this audio accurately.

Return ONLY valid JSON.

Use exactly this structure:

{
  "language": "detected language",
  "segments": [
    {
      "start": 0.0,
      "end": 0.0,
      "text": "spoken text"
    }
  ]
}

Rules:

1. Transcribe only speech that is actually present in the audio.
2. Do not invent or infer missing words.
3. Preserve the spoken wording as accurately as possible.
4. Use seconds for timestamps.
5. Each segment must contain a start and end timestamp.
6. Keep timestamps in chronological order.
7. Do not assign speaker names.
8. Do not add explanations outside the JSON.
"""

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[
                audio_file,
                prompt,
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
            ),
        )

    except errors.ServerError as exc:
        if exc.code == 503:
            raise GeminiUnavailableError(
                "Gemini is temporarily unavailable. "
                "Please try the analysis again."
            ) from exc

        raise RuntimeError(
            f"Gemini server error: {exc}"
        ) from exc

    except errors.APIError as exc:
        raise RuntimeError(
            f"Gemini API error: {exc}"
        ) from exc

    if not response.text:
        raise RuntimeError(
            "Gemini returned an empty transcription response."
        )

    response_text = response.text.strip()

    try:
        result = json.loads(response_text)

    except json.JSONDecodeError:

        if response_text.startswith("```"):
            lines = response_text.splitlines()

            if (
                len(lines) >= 3
                and lines[0].startswith("```")
                and lines[-1].strip() == "```"
            ):
                response_text = "\n".join(
                    lines[1:-1]
                ).strip()

                try:
                    result = json.loads(
                        response_text
                    )

                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        "Gemini returned malformed JSON."
                    ) from exc

            else:
                raise RuntimeError(
                    "Gemini returned malformed JSON."
                )

        else:
            raise RuntimeError(
                "Gemini returned malformed JSON."
            )

    return result


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
        f"Model: {MODEL_NAME}"
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

    except GeminiUnavailableError as exc:
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
