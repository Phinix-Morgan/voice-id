import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier


MODEL_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"

DEVICE = (
    "cuda:0"
    if torch.cuda.is_available()
    else "cpu"
)

WINDOW_SECONDS = 2.0
STEP_SECONDS = 1.0

# Initial experimental threshold.
# We will NOT treat this as a final diarization threshold.
CHANGE_THRESHOLD = 0.50


def load_model() -> EncoderClassifier:
    """Load ECAPA-TDNN."""

    print(f"Using device: {DEVICE}")

    model = EncoderClassifier.from_hparams(
        source=MODEL_SOURCE,
        run_opts={"device": DEVICE},
    )

    print("ECAPA-TDNN loaded successfully.")

    return model


def load_audio(
    audio_path: Path,
) -> tuple[torch.Tensor, int]:
    """Load audio and convert it to mono 16 kHz."""

    signal, sample_rate = torchaudio.load(
        str(audio_path)
    )

    # Stereo → mono
    if signal.shape[0] > 1:
        signal = signal.mean(
            dim=0,
            keepdim=True,
        )

    # Resample → 16 kHz
    if sample_rate != 16000:

        resampler = (
            torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=16000,
            )
        )

        signal = resampler(signal)

        sample_rate = 16000

    return signal, sample_rate


def extract_embedding(
    model: EncoderClassifier,
    segment: torch.Tensor,
) -> torch.Tensor:
    """Extract one ECAPA embedding."""

    segment = segment.to(DEVICE)

    with torch.no_grad():

        embedding = model.encode_batch(
            segment
        )

    embedding = embedding.squeeze()

    embedding = F.normalize(
        embedding.unsqueeze(0),
        p=2,
        dim=1,
    ).squeeze(0)

    return embedding


def cosine_similarity(
    first: torch.Tensor,
    second: torch.Tensor,
) -> float:
    """Calculate cosine similarity."""

    return torch.dot(
        first,
        second,
    ).item()


def create_windows(
    signal: torch.Tensor,
    sample_rate: int,
) -> list[tuple[float, float, torch.Tensor]]:
    """
    Create overlapping audio windows.

    Returns:
        [
            (start_seconds, end_seconds, audio_tensor),
            ...
        ]
    """

    window_size = int(
        WINDOW_SECONDS * sample_rate
    )

    step_size = int(
        STEP_SECONDS * sample_rate
    )

    total_samples = signal.shape[-1]

    windows = []

    start = 0

    while (
        start + window_size
        <= total_samples
    ):

        end = start + window_size

        start_seconds = (
            start / sample_rate
        )

        end_seconds = (
            end / sample_rate
        )

        segment = signal[
            :,
            start:end,
        ]

        windows.append(
            (
                start_seconds,
                end_seconds,
                segment,
            )
        )

        start += step_size

    return windows


def analyze_turns(
    model: EncoderClassifier,
    windows,
) -> None:
    """Analyze similarity between consecutive windows."""

    print()
    print("=" * 70)
    print("SPEAKER TURN ANALYSIS")
    print("=" * 70)

    embeddings = []

    print(
        f"\nWindows: {len(windows)}"
    )

    print(
        f"Window size: "
        f"{WINDOW_SECONDS:.1f}s"
    )

    print(
        f"Step size: "
        f"{STEP_SECONDS:.1f}s"
    )

    # -----------------------------------------------------
    # Extract embeddings
    # -----------------------------------------------------

    for index, (
        start,
        end,
        segment,
    ) in enumerate(
        windows,
        start=1,
    ):

        print(
            f"\n[{index}] "
            f"{start:.2f}s → {end:.2f}s"
        )

        embedding = extract_embedding(
            model,
            segment,
        )

        embeddings.append(
            embedding
        )

        print(
            f"  Embedding: "
            f"{embedding.shape}"
        )

    # -----------------------------------------------------
    # Compare consecutive windows
    # -----------------------------------------------------

    print()
    print("=" * 70)
    print("CONSECUTIVE WINDOW SIMILARITY")
    print("=" * 70)

    if len(embeddings) < 2:

        print(
            "\nNot enough windows "
            "for turn analysis."
        )

        return

    for index in range(
        len(embeddings) - 1
    ):

        first_start, first_end, _ = (
            windows[index]
        )

        second_start, second_end, _ = (
            windows[index + 1]
        )

        similarity = cosine_similarity(
            embeddings[index],
            embeddings[index + 1],
        )

        possible_change = (
            similarity < CHANGE_THRESHOLD
        )

        marker = (
            "  ← POSSIBLE SPEAKER CHANGE"
            if possible_change
            else ""
        )

        print(
            f"\n"
            f"{first_start:.2f}–{first_end:.2f}s"
            f"  ↔  "
            f"{second_start:.2f}–{second_end:.2f}s"
        )

        print(
            f"Similarity: "
            f"{similarity:.4f}"
            f"{marker}"
        )


def main() -> None:

    print("=" * 70)
    print("SPEAKER TURN DETECTOR")
    print("=" * 70)

    if len(sys.argv) != 2:

        print(
            "\nUsage:"
        )

        print(
            "  uv run python -m "
            "voice_id.turns <audio.wav>"
        )

        raise SystemExit(1)

    audio_path = Path(
        sys.argv[1]
    )

    if not audio_path.exists():

        raise FileNotFoundError(
            f"Audio file not found: "
            f"{audio_path}"
        )

    print(
        f"\nAudio: {audio_path}"
    )

    model = load_model()

    print("\nLoading audio...")

    signal, sample_rate = load_audio(
        audio_path
    )

    duration = (
        signal.shape[-1]
        / sample_rate
    )

    print(
        f"Sample rate: "
        f"{sample_rate} Hz"
    )

    print(
        f"Duration: "
        f"{duration:.2f}s"
    )

    windows = create_windows(
        signal,
        sample_rate,
    )

    if not windows:

        print(
            "\nAudio is shorter than "
            f"{WINDOW_SECONDS:.1f}s."
        )

        return

    analyze_turns(
        model,
        windows,
    )


if __name__ == "__main__":
    main()
