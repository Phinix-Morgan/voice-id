from __future__ import annotations

import sys
import uuid
from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request
from werkzeug.utils import secure_filename

import os

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent

SRC_DIR = ROOT_DIR / "src"
UPLOAD_DIR = ROOT_DIR / "data" / "evaluation" / "uploads"

UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from voice_id.audio import normalize_audio  # noqa: E402
from voice_id.pipeline import process_audio  # noqa: E402
from voice_id.notes import (  # noqa: E402
    NotesContentError,
    NotesError,
    generate_notes,
)
from voice_id.transcription import VoskUnavailableError


# ---------------------------------------------------------------------------
# Flask application
# ---------------------------------------------------------------------------

app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024


ALLOWED_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".aac",
    ".m4a",
    ".flac",
    ".ogg",
    ".webm",
}


# ---------------------------------------------------------------------------
# Analysis store
#
# In-memory ticket system: analysis_id -> {"result", "notes"}.
# "result" is the pipeline output; "notes" is the cached generated
# notes (None until first requested).
#
# This dict is the single swap point for future persistence —
# replacing it with a disk/database store changes nothing else.
# ---------------------------------------------------------------------------

ANALYSES: dict[str, dict] = {}

MAX_STORED_ANALYSES = 20


def store_analysis(result: dict) -> str:

    # Evict the oldest analysis when the cap is reached
    if len(ANALYSES) >= MAX_STORED_ANALYSES:
        ANALYSES.pop(next(iter(ANALYSES)))

    analysis_id = uuid.uuid4().hex

    ANALYSES[analysis_id] = {
        "result": result,
        "notes": None,
    }

    return analysis_id


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def allowed_file(filename: str) -> bool:
    suffix = Path(filename).suffix.lower()

    return suffix in ALLOWED_EXTENSIONS


def error_response(
    message: str,
    status_code: int,
):
    return jsonify(
        {
            "success": False,
            "error": message,
        }
    ), status_code


def format_timestamp(seconds) -> str:

    try:
        total = int(round(float(seconds)))

    except (TypeError, ValueError):
        return ""

    minutes, secs = divmod(total, 60)

    return f"{minutes:02d}:{secs:02d}"


def notes_to_markdown(notes: dict) -> str:

    lines = []

    lines.append(f"# {notes.get('title', 'Session notes')}")
    lines.append("")

    if notes.get("mode"):
        lines.append(f"**Mode:** {notes['mode']}")
        lines.append("")

    if notes.get("tldr"):
        lines.append("## Summary")
        lines.append("")
        lines.append(notes["tldr"])
        lines.append("")

    topics = notes.get("topics") or []

    if topics:
        lines.append("## Topics")
        lines.append("")

        for topic in topics:
            start = format_timestamp(topic.get("start"))
            end = format_timestamp(topic.get("end"))

            lines.append(f"### {topic.get('name', 'Untitled')} ({start} – {end})")
            lines.append("")

            if topic.get("summary"):
                lines.append(topic["summary"])
                lines.append("")

            for point in topic.get("key_points") or []:
                lines.append(f"- {point}")

            lines.append("")

    concepts = notes.get("concepts") or []

    if concepts:
        lines.append("## Key Concepts")
        lines.append("")

        for concept in concepts:
            lines.append(
                f"**{concept.get('term', '')}** — {concept.get('definition', '')}"
            )

            stamps = ", ".join(
                format_timestamp(ts)
                for ts in concept.get("timestamps") or []
            )

            if stamps:
                lines.append(f"  *(mentioned at {stamps})*")

            lines.append("")

    contributions = notes.get("speaker_contributions") or []

    if contributions:
        lines.append("## Speaker Contributions")
        lines.append("")

        for contribution in contributions:
            lines.append(f"### {contribution.get('speaker', 'unknown')}")
            lines.append("")

            for point in contribution.get("points") or []:
                lines.append(f"- {point}")

            lines.append("")

    action_items = notes.get("action_items") or []

    if action_items:
        lines.append("## Action Items")
        lines.append("")

        for item in action_items:
            owner = item.get("owner")
            owner_text = f" *(owner: {owner})*" if owner else ""

            stamp = format_timestamp(item.get("timestamp"))
            stamp_text = f" `[{stamp}]`" if stamp else ""

            lines.append(f"- [ ] {item.get('task', '')}{owner_text}{stamp_text}")

        lines.append("")

    diagram = notes.get("diagram")

    if diagram:
        lines.append("## Diagram")
        lines.append("")
        lines.append("```mermaid")
        lines.append(diagram)
        lines.append("```")
        lines.append("")

    return "\n".join(lines).strip() + "\n"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/analyze")
def analyze_audio():

    if "audio" not in request.files:
        return error_response(
            "No audio file was uploaded.",
            400,
        )

    uploaded_file = request.files["audio"]

    if not uploaded_file.filename:
        return error_response(
            "No audio file was selected.",
            400,
        )

    if not allowed_file(
        uploaded_file.filename
    ):
        return error_response(
            (
                "Unsupported audio format. "
                "Supported formats: WAV, MP3, AAC, M4A, "
                "FLAC, OGG and WEBM."
            ),
            400,
        )

    original_name = secure_filename(
        uploaded_file.filename
    )

    unique_name = (
        f"{uuid.uuid4().hex}_"
        f"{original_name}"
    )

    audio_path = UPLOAD_DIR / unique_name

    normalized_path = None

    try:

        # ---------------------------------------------------------------
        # 1. Save uploaded file
        # ---------------------------------------------------------------

        uploaded_file.save(
            audio_path
        )

        # ---------------------------------------------------------------
        # 2. Normalize audio
        #
        # Every supported input format is converted into the format
        # expected by the ML pipeline.
        # ---------------------------------------------------------------

        normalized_path = normalize_audio(
            audio_path
        )

        # ---------------------------------------------------------------
        # 3. Run complete Voice ID pipeline
        #
        # Diarization
        # → transcription
        # → alignment
        # ---------------------------------------------------------------

        result = process_audio(
            normalized_path
        )

        # ---------------------------------------------------------------
        # 4. Store result and issue analysis_id ticket
        #
        # The ticket lets /api/notes and the export route reference
        # this analysis without the browser re-sending anything.
        # ---------------------------------------------------------------

        analysis_id = store_analysis(
            result
        )

        return jsonify(
            {
                "success": True,
                "analysis_id": analysis_id,
                "result": result,
            }
        )

    except VoskUnavailableError:

        app.logger.warning(
            "Vosk transcription service is temporarily unavailable."
        )

        return error_response(
            (
                "Transcription is temporarily unavailable. "
                "Please try the analysis again."
            ),
            503,
        )

    except FileNotFoundError as exc:

        app.logger.warning(
            "Audio file error: %s",
            exc,
        )

        return error_response(
            str(exc),
            400,
        )

    except ValueError as exc:

        app.logger.warning(
            "Audio validation error: %s",
            exc,
        )

        return error_response(
            str(exc),
            400,
        )

    except RuntimeError as exc:

        app.logger.exception(
            "Audio processing failed."
        )

        return error_response(
            str(exc),
            500,
        )

    except Exception:

        app.logger.exception(
            "Unexpected audio processing failure."
        )

        return error_response(
            (
                "The audio could not be analyzed. "
                "Please try again."
            ),
            500,
        )

    finally:

        # ---------------------------------------------------------------
        # Remove uploaded source file
        # ---------------------------------------------------------------

        if audio_path.exists():
            audio_path.unlink()

        # ---------------------------------------------------------------
        # Remove normalized temporary WAV
        # ---------------------------------------------------------------

        if normalized_path is not None:
            normalized_path = Path(
                normalized_path
            )

            if normalized_path.exists():
                normalized_path.unlink()


@app.post("/api/notes")
def create_notes():

    payload = request.get_json(
        silent=True
    )

    if not isinstance(payload, dict) or not payload.get("analysis_id"):
        return error_response(
            "No analysis_id was provided.",
            400,
        )

    analysis_id = payload["analysis_id"]

    record = ANALYSES.get(
        analysis_id
    )

    if record is None:
        return error_response(
            "Unknown analysis. Please analyze the audio again.",
            404,
        )

    # ---------------------------------------------------------------
    # 1. Serve cached notes when they already exist
    # ---------------------------------------------------------------

    if record["notes"] is not None:
        return jsonify(
            {
                "success": True,
                "cached": True,
                "notes": record["notes"],
            }
        )

    # ---------------------------------------------------------------
    # 2. Generate and validate notes from the stored result
    # ---------------------------------------------------------------

    try:

        notes = generate_notes(
            record["result"]
        )

    except NotesContentError as exc:

        app.logger.warning(
            "Notes content error: %s",
            exc,
        )

        return error_response(
            str(exc),
            400,
        )

    except NotesError as exc:

        app.logger.warning(
            "Notes generation failed: %s",
            exc,
        )

        return error_response(
            (
                "Notes could not be generated right now. "
                "Please try again shortly."
            ),
            503,
        )

    except Exception:

        app.logger.exception(
            "Unexpected notes generation failure."
        )

        return error_response(
            "Notes could not be generated. Please try again.",
            500,
        )

    # ---------------------------------------------------------------
    # 3. Cache for repeat views and export
    # ---------------------------------------------------------------

    record["notes"] = notes

    return jsonify(
        {
            "success": True,
            "cached": False,
            "notes": notes,
        }
    )


@app.get("/api/notes/<analysis_id>/export")
def export_notes(analysis_id):

    record = ANALYSES.get(
        analysis_id
    )

    if record is None:
        return error_response(
            "Unknown analysis.",
            404,
        )

    if record["notes"] is None:
        return error_response(
            "Notes have not been generated yet.",
            404,
        )

    markdown = notes_to_markdown(
        record["notes"]
    )

    return Response(
        markdown,
        mimetype="text/markdown",
        headers={
            "Content-Disposition": (
                f"attachment; filename=voice-id-notes-{analysis_id}.md"
            ),
        },
    )


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------

@app.errorhandler(413)
def request_entity_too_large(_error):

    return error_response(
        (
            "The uploaded file is too large. "
            "Maximum size is 200 MB."
        ),
        413,
    )


# ---------------------------------------------------------------------------
# Development entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False,
        use_reloader=False,
    )
