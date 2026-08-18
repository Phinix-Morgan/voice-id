import sys
from pathlib import Path

import torch
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier


MODEL_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
DATA_DIR = Path("data/processed")


def load_model() -> EncoderClassifier:
    """Load the pretrained ECAPA-TDNN speaker encoder."""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")

    return EncoderClassifier.from_hparams(
        source=MODEL_SOURCE,
        run_opts={"device": device},
    )


def extract_embedding(
    model: EncoderClassifier,
    audio_path: str | Path,
) -> torch.Tensor:
    """Extract a speaker embedding from an audio file."""
    signal, sample_rate = torchaudio.load(str(audio_path))

    # Convert stereo to mono.
    if signal.shape[0] > 1:
        signal = signal.mean(dim=0, keepdim=True)

    # ECAPA-TDNN expects 16 kHz audio.
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate,
            new_freq=16000,
        )
        signal = resampler(signal)

    # Move audio to the model's device.
    signal = signal.to(model.device)

    with torch.no_grad():
        embedding = model.encode_batch(signal)

    # Keep the embedding on the GPU.
    return embedding.squeeze()


def main() -> None:
    if len(sys.argv) != 3:
        print(
            "Usage: "
            "uv run python -m voice_id.embeddings "
            "<dataset> <speaker>"
        )

        print("\nExamples:")
        print(
            "  uv run python -m voice_id.embeddings known shiv"
        )
        print(
            "  uv run python -m voice_id.embeddings known friend"
        )
        print(
            "  uv run python -m voice_id.embeddings "
            "evaluation speaker_02"
        )

        raise SystemExit(1)

    dataset = sys.argv[1]
    speaker = sys.argv[2]

    valid_datasets = {"known", "evaluation"}

    if dataset not in valid_datasets:
        raise ValueError(
            f"Invalid dataset '{dataset}'. "
            f"Choose from: {', '.join(sorted(valid_datasets))}"
        )

    audio_dir = DATA_DIR / dataset / speaker
    embeddings_dir = audio_dir / "embeddings"

    if not audio_dir.exists():
        raise FileNotFoundError(
            f"Audio directory not found: {audio_dir}"
        )

    embeddings_dir.mkdir(parents=True, exist_ok=True)

    print("Loading ECAPA-TDNN...")

    model = load_model()

    print("Model loaded successfully.")
    print(f"Model device: {model.device}")

    audio_files = sorted(audio_dir.glob("*.wav"))

    if not audio_files:
        raise FileNotFoundError(
            f"No WAV files found in {audio_dir}"
        )

    print(f"\nDataset: {dataset}")
    print(f"Speaker: {speaker}")
    print(f"Found {len(audio_files)} WAV files.")

    for index, audio_path in enumerate(audio_files, start=1):
        print(
            f"\n[{index}/{len(audio_files)}] "
            f"Processing: {audio_path.name}"
        )

        embedding = extract_embedding(
            model=model,
            audio_path=audio_path,
        )

        output_path = (
            embeddings_dir / f"{audio_path.stem}.pt"
        )

        torch.save(embedding, output_path)

        print(f"  Shape:  {embedding.shape}")
        print(f"  Device: {embedding.device}")
        print(f"  Saved:  {output_path}")

    print(
        f"\nAll embeddings for '{speaker}' "
        f"in '{dataset}' generated successfully."
    )


if __name__ == "__main__":
    main()
