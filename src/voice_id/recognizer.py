from pathlib import Path

import torch
import torch.nn.functional as F
from speechbrain.inference.speaker import EncoderClassifier


MODEL_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"

CLASSIFIER_PATH = Path(
    "data/models/speaker_classifier.pt"
)

KNOWN_DIR = Path(
    "data/processed/known"
)

EVALUATION_DIR = Path(
    "data/processed/evaluation"
)

DEVICE = (
    "cuda:0"
    if torch.cuda.is_available()
    else "cpu"
)

# ---------------------------------------------------------
# Unknown-speaker rejection thresholds
# ---------------------------------------------------------

CLASSIFIER_CONFIDENCE_THRESHOLD = 0.80

SIMILARITY_THRESHOLD = 0.60


def load_ecapa() -> EncoderClassifier:
    """Load the pretrained ECAPA-TDNN speaker encoder."""

    print(f"Using device: {DEVICE}")

    return EncoderClassifier.from_hparams(
        source=MODEL_SOURCE,
        run_opts={"device": DEVICE},
    )


def load_classifier():
    """Load the trained speaker classifier."""

    if not CLASSIFIER_PATH.exists():
        raise FileNotFoundError(
            f"Classifier not found: {CLASSIFIER_PATH}"
        )

    checkpoint = torch.load(
        CLASSIFIER_PATH,
        map_location="cpu",
        weights_only=False,
    )

    return (
        checkpoint["classifier"],
        checkpoint["speakers"],
    )


def load_known_profiles(
    speakers: list[str],
) -> dict[str, torch.Tensor]:
    """
    Build one profile embedding for each known speaker.

    The profile is the normalized mean of that speaker's
    training embeddings.
    """

    profiles = {}

    for speaker in speakers:
        embeddings_dir = (
            KNOWN_DIR
            / speaker
            / "embeddings"
        )

        files = sorted(
            embeddings_dir.glob("*.pt")
        )

        if not files:
            raise FileNotFoundError(
                f"No embeddings found for {speaker}: "
                f"{embeddings_dir}"
            )

        embeddings = []

        for path in files:
            embedding = torch.load(
                path,
                map_location=DEVICE,
                weights_only=True,
            )

            embeddings.append(
                embedding.to(DEVICE)
            )

        matrix = torch.stack(embeddings)

        profile = matrix.mean(dim=0)

        profile = F.normalize(
            profile.unsqueeze(0),
            p=2,
            dim=1,
        ).squeeze(0)

        profiles[speaker] = profile

        print(
            f"{speaker}: "
            f"{len(files)} embeddings → "
            f"profile {profile.shape}"
        )

    return profiles


def extract_embedding(
    model: EncoderClassifier,
    audio_path: Path,
) -> torch.Tensor:
    """Extract a 192-D ECAPA embedding."""

    import torchaudio

    signal, sample_rate = torchaudio.load(
        str(audio_path)
    )

    if signal.shape[0] > 1:
        signal = signal.mean(
            dim=0,
            keepdim=True,
        )

    if sample_rate != 16000:
        resampler = (
            torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=16000,
            )
        )

        signal = resampler(signal)

    signal = signal.to(DEVICE)

    with torch.no_grad():
        embedding = model.encode_batch(
            signal
        )

    embedding = embedding.squeeze()

    return embedding


def cosine_similarity(
    embedding: torch.Tensor,
    profile: torch.Tensor,
) -> float:
    """Calculate cosine similarity."""

    embedding = F.normalize(
        embedding.unsqueeze(0),
        p=2,
        dim=1,
    )

    profile = F.normalize(
        profile.unsqueeze(0),
        p=2,
        dim=1,
    )

    similarity = (
        embedding @ profile.T
    ).item()

    return similarity


def classify_embedding(
    classifier,
    embedding: torch.Tensor,
) -> tuple[str, float]:
    """
    Classify an embedding and return:

        predicted speaker
        classifier confidence
    """

    embedding_cpu = (
        embedding
        .detach()
        .cpu()
        .numpy()
        .reshape(1, -1)
    )

    probabilities = (
        classifier.predict_proba(
            embedding_cpu
        )[0]
    )

    index = probabilities.argmax()

    speaker = classifier.classes_[index]

    confidence = float(
        probabilities[index]
    )

    return speaker, confidence


def recognize(
    classifier,
    profiles,
    model,
    audio_path: Path,
) -> dict:
    """
    Recognize a single audio file.

    Decision:

        confident classifier
              +
        sufficient similarity
              ↓
             known

        otherwise
              ↓
           unknown
    """

    embedding = extract_embedding(
        model,
        audio_path,
    )

    predicted_speaker, confidence = (
        classify_embedding(
            classifier,
            embedding,
        )
    )

    similarities = {}

    for speaker, profile in profiles.items():
        similarities[speaker] = (
            cosine_similarity(
                embedding,
                profile,
            )
        )

    best_similarity_speaker = max(
        similarities,
        key=similarities.get,
    )

    best_similarity = similarities[
        best_similarity_speaker
    ]

    is_known = (
        confidence
        >= CLASSIFIER_CONFIDENCE_THRESHOLD
        and best_similarity
        >= SIMILARITY_THRESHOLD
    )

    if is_known:
        identity = predicted_speaker
    else:
        identity = "unknown"

    return {
        "identity": identity,
        "predicted_speaker": predicted_speaker,
        "confidence": confidence,
        "similarities": similarities,
        "best_similarity_speaker":
            best_similarity_speaker,
        "best_similarity":
            best_similarity,
    }


def main() -> None:

    print("=" * 70)
    print("VOICE RECOGNIZER")
    print("=" * 70)

    print("\nLoading classifier...")

    classifier, speakers = (
        load_classifier()
    )

    print(
        f"Known speakers: "
        f"{', '.join(speakers)}"
    )

    print("\nLoading ECAPA-TDNN...")

    model = load_ecapa()

    print("Model loaded successfully.")

    print("\nBuilding speaker profiles...")

    profiles = load_known_profiles(
        speakers
    )

    print("\nEnrollment complete.")

    # -----------------------------------------------------
    # Evaluation data
    # -----------------------------------------------------

    evaluation_speakers = sorted(
        [
            path.name
            for path in EVALUATION_DIR.iterdir()
            if path.is_dir()
        ]
    )

    if not evaluation_speakers:
        raise FileNotFoundError(
            f"No evaluation speakers found in "
            f"{EVALUATION_DIR}"
        )

    print()
    print("=" * 70)
    print("UNKNOWN-SPEAKER EVALUATION")
    print("=" * 70)

    for speaker in evaluation_speakers:

        audio_dir = (
            EVALUATION_DIR / speaker
        )

        audio_files = sorted(
            audio_dir.glob("*.wav")
        )

        if not audio_files:
            print(
                f"\nNo WAV files found for "
                f"{speaker}"
            )
            continue

        print()
        print(
            f"[EVALUATION: {speaker}]"
        )

        for audio_path in audio_files:

            result = recognize(
                classifier,
                profiles,
                model,
                audio_path,
            )

            print()
            print(audio_path.stem)

            print(
                f"  Classifier : "
                f"{result['predicted_speaker']}"
            )

            print(
                f"  Confidence : "
                f"{result['confidence']:.4f}"
            )

            print(
                f"  Shiv       : "
                f"{result['similarities'].get('shiv', 0.0):.4f}"
            )

            print(
                f"  Friend     : "
                f"{result['similarities'].get('friend', 0.0):.4f}"
            )

            print(
                f"  Best match : "
                f"{result['best_similarity_speaker']}"
            )

            print(
                f"  Similarity : "
                f"{result['best_similarity']:.4f}"
            )

            print(
                f"  FINAL      : "
                f"{result['identity']}"
            )


if __name__ == "__main__":
    main()
