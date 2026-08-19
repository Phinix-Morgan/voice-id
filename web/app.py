from __future__ import annotations

import sys
import uuid
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename


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
from voice_id.transcription import (  # noqa: E402
    GeminiUnavailableError,
)


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

        return jsonify(
            {
                "success": True,
                "result": result,
            }
        )

    except GeminiUnavailableError:

        app.logger.warning(
            "Gemini transcription service is temporarily unavailable."
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
        port=5000,
        debug=False,
        use_reloader=False,
    )
