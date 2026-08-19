FROM python:3.12-slim

# --------------------------------------------------------------------------
# System dependencies
# --------------------------------------------------------------------------

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        libgomp1 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*


# --------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PORT=5000 \
    VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"


# --------------------------------------------------------------------------
# Application
# --------------------------------------------------------------------------

WORKDIR /app


# --------------------------------------------------------------------------
# Install uv
# --------------------------------------------------------------------------

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/


# --------------------------------------------------------------------------
# Create CPU-only virtual environment
# --------------------------------------------------------------------------

RUN uv venv /app/.venv --python 3.12


# --------------------------------------------------------------------------
# Install CPU PyTorch stack
# --------------------------------------------------------------------------

RUN uv pip install \
        --python /app/.venv/bin/python \
        --index-url https://download.pytorch.org/whl/cpu \
        "torch==2.13.0" \
        "torchaudio==2.11.0" \
        "torchcodec==0.16.0"


# --------------------------------------------------------------------------
# Project dependencies
# --------------------------------------------------------------------------

COPY pyproject.toml README.md ./

RUN uv pip install \
        --python /app/.venv/bin/python \
        "flask>=3.1.3" \
        "google-genai>=2.18.1" \
        "matplotlib>=3.11.1" \
        "numpy>=2.5.2" \
        "python-dotenv>=1.2.3" \
        "scikit-learn>=1.9.0" \
        "sounddevice>=0.5.5" \
        "soundfile>=0.14.0" \
        "speechbrain>=1.1.0"


# --------------------------------------------------------------------------
# Application source
# --------------------------------------------------------------------------

COPY src ./src
COPY web ./web

COPY data/processed/known ./data/processed/known


# --------------------------------------------------------------------------
# Install Voice ID itself
# --------------------------------------------------------------------------

RUN uv pip install \
        --python /app/.venv/bin/python \
        --no-deps \
        .


# --------------------------------------------------------------------------
# Runtime
# --------------------------------------------------------------------------

EXPOSE 5000

CMD ["python", "web/app.py"]
