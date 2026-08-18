from pathlib import Path

import torch
import torch.nn.functional as F


BASE_DIR = Path("data/processed")

SPEAKERS = {
    "shiv": BASE_DIR / "shiv" / "embeddings",
    "speaker_02": BASE_DIR / "speaker_02" / "embeddings",
}

ENROLLMENT_SIZE = 10


def load_embeddings(
    embeddings_dir: Path,
) -> tuple[list[str], torch.Tensor]:
    """Load all embeddings for one speaker onto the GPU."""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    files = sorted(embeddings_dir.glob("*.pt"))

    if not files:
        raise FileNotFoundError(
            f"No embeddings found in {embeddings_dir}"
        )

    names = []
    embeddings = []

    for path in files:
        embedding = torch.load(
            path,
            map_location=device,
            weights_only=True,
        )

        names.append(path.stem)
        embeddings.append(embedding)

    return names, torch.stack(embeddings)


def create_profile(
    embeddings: torch.Tensor,
) -> torch.Tensor:
    """Create a normalized speaker centroid."""
    centroid = embeddings.mean(dim=0)

    return F.normalize(
        centroid.unsqueeze(0),
        p=2,
        dim=1,
    ).squeeze(0)


def cosine_similarity(
    embedding: torch.Tensor,
    profile: torch.Tensor,
) -> torch.Tensor:
    """Calculate cosine similarity between an embedding and profile."""
    embedding = F.normalize(
        embedding.unsqueeze(0),
        p=2,
        dim=1,
    ).squeeze(0)

    return torch.dot(embedding, profile)


def predict_speaker(
    embedding: torch.Tensor,
    profiles: dict[str, torch.Tensor],
) -> tuple[str, dict[str, float]]:
    """Predict which enrolled speaker matches the embedding."""
    scores = {}

    for speaker, profile in profiles.items():
        score = cosine_similarity(
            embedding,
            profile,
        )

        scores[speaker] = score.item()

    predicted = max(
        scores,
        key=scores.get,
    )

    return predicted, scores


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")

    # ---------------------------------------------------------
    # Load embeddings
    # ---------------------------------------------------------

    speaker_data = {}

    for speaker, directory in SPEAKERS.items():
        names, embeddings = load_embeddings(directory)

        if len(embeddings) <= ENROLLMENT_SIZE:
            raise ValueError(
                f"{speaker} needs more than "
                f"{ENROLLMENT_SIZE} recordings."
            )

        speaker_data[speaker] = {
            "names": names,
            "embeddings": embeddings,
        }

        print(
            f"{speaker}: "
            f"{embeddings.shape} "
            f"on {embeddings.device}"
        )

    # ---------------------------------------------------------
    # Create enrollment profiles
    # ---------------------------------------------------------

    profiles = {}

    print("\n" + "=" * 70)
    print("ENROLLMENT")
    print("=" * 70)

    for speaker, data in speaker_data.items():
        enrollment = data["embeddings"][:ENROLLMENT_SIZE]

        profile = create_profile(enrollment)

        profiles[speaker] = profile

        print(
            f"{speaker}: "
            f"{ENROLLMENT_SIZE} recordings → "
            f"profile {profile.shape} "
            f"on {profile.device}"
        )

    # ---------------------------------------------------------
    # Test unseen recordings
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("UNSEEN TEST RESULTS")
    print("=" * 70)

    total = 0
    correct = 0

    for expected_speaker, data in speaker_data.items():
        test_embeddings = data["embeddings"][ENROLLMENT_SIZE:]
        test_names = data["names"][ENROLLMENT_SIZE:]

        print(f"\n[{expected_speaker.upper()} TEST SET]")

        for name, embedding in zip(
            test_names,
            test_embeddings,
        ):
            predicted, scores = predict_speaker(
                embedding,
                profiles,
            )

            total += 1

            if predicted == expected_speaker:
                correct += 1
                result = "✓"
            else:
                result = "✗"

            print(f"\n{name}")
            print(
                f"  Shiv       : "
                f"{scores['shiv']:.4f}"
            )
            print(
                f"  Speaker 02 : "
                f"{scores['speaker_02']:.4f}"
            )
            print(
                f"  Prediction : "
                f"{predicted}"
            )
            print(
                f"  Expected   : "
                f"{expected_speaker}"
            )
            print(
                f"  Result     : "
                f"{result}"
            )

    # ---------------------------------------------------------
    # Accuracy
    # ---------------------------------------------------------

    accuracy = correct / total

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    print(f"\nCorrect   : {correct} / {total}")
    print(f"Accuracy  : {accuracy:.2%}")


if __name__ == "__main__":
    main()
