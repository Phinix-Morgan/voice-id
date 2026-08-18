import sys
from pathlib import Path

import torch
from speechbrain.inference.VAD import VAD


DEVICE = (
    "cuda:0"
    if torch.cuda.is_available()
    else "cpu"
)

VAD_SOURCE = "speechbrain/vad-crdnn-libriparty"

MIN_SPEECH_DURATION = 0.50
MERGE_GAP = 0.30


def load_vad() -> VAD:
    """Load the SpeechBrain voice activity detector."""

    print(f"Using device: {DEVICE}")

    vad = VAD.from_hparams(
        source=VAD_SOURCE,
        savedir="pretrained_models/vad",
        run_opts={"device": DEVICE},
    )

    print("VAD loaded successfully.")

    return vad


def segment_audio(
    vad: VAD,
    audio_path: Path,
) -> list[tuple[float, float]]:
    """
    Detect speech regions.

    Returns:
        [
            (start_seconds, end_seconds),
            ...
        ]
    """

    print(f"\nAnalyzing: {audio_path}")

    boundaries = vad.get_speech_segments(
        str(audio_path)
    )

    segments = []

    for boundary in boundaries:

        start = float(boundary[0])
        end = float(boundary[1])

        duration = end - start

        if duration < MIN_SPEECH_DURATION:
            continue

        segments.append(
            (start, end)
        )

    return segments


def merge_segments(
    segments: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """
    Merge speech segments separated by
    a very short silence.
    """

    if not segments:
        return []

    merged = [segments[0]]

    for start, end in segments[1:]:

        previous_start, previous_end = merged[-1]

        gap = start - previous_end

        if gap <= MERGE_GAP:

            merged[-1] = (
                previous_start,
                end,
            )

        else:

            merged.append(
                (start, end)
            )

    return merged


def main() -> None:

    print("=" * 70)
    print("SPEECH SEGMENTATION")
    print("=" * 70)

    if len(sys.argv) != 2:

        print(
            "\nUsage:"
        )

        print(
            "  uv run python -m "
            "voice_id.segmentation <audio.wav>"
        )

        print(
            "\nExample:"
        )

        print(
            '  uv run python -m voice_id.segmentation '
            '"data/processed/known/shiv/audio.wav"'
        )

        raise SystemExit(1)

    audio_path = Path(sys.argv[1])

    if not audio_path.exists():

        raise FileNotFoundError(
            f"Audio file not found: "
            f"{audio_path}"
        )

    vad = load_vad()

    segments = segment_audio(
        vad,
        audio_path,
    )

    segments = merge_segments(
        segments
    )

    print()
    print("=" * 70)
    print("DETECTED SPEECH SEGMENTS")
    print("=" * 70)

    if not segments:

        print("\nNo speech detected.")

        return

    total_speech = 0.0

    for index, (start, end) in enumerate(
        segments,
        start=1,
    ):

        duration = end - start

        total_speech += duration

        print(
            f"\n[{index}]"
        )

        print(
            f"  Start    : {start:.2f}s"
        )

        print(
            f"  End      : {end:.2f}s"
        )

        print(
            f"  Duration : {duration:.2f}s"
        )

    print()
    print("=" * 70)

    print(
        f"Segments       : {len(segments)}"
    )

    print(
        f"Speech duration: "
        f"{total_speech:.2f}s"
    )


if __name__ == "__main__":
    main()
