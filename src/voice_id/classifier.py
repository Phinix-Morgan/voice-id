from pathlib import Path

import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


KNOWN_DIR = Path("data/processed/known")

TRAIN_PER_SPEAKER = 10
TEST_PER_SPEAKER = 5

RANDOM_STATE = 42

CLASSIFIER_PATH = Path("data/models/speaker_classifier.pt")


def load_embeddings(
    speaker: str,
) -> tuple[list[str], torch.Tensor]:
    """Load all ECAPA embeddings for a speaker."""

    embeddings_dir = KNOWN_DIR / speaker / "embeddings"

    if not embeddings_dir.exists():
        raise FileNotFoundError(
            f"Embeddings directory not found: {embeddings_dir}"
        )

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
            map_location="cpu",
            weights_only=True,
        )

        names.append(path.stem)
        embeddings.append(embedding)

    return names, torch.stack(embeddings)


def split_speaker_data(
    names: list[str],
    embeddings: torch.Tensor,
    speaker: str,
) -> tuple:
    """Split one speaker's embeddings into train and test sets."""

    if len(embeddings) != TRAIN_PER_SPEAKER + TEST_PER_SPEAKER:
        raise ValueError(
            f"{speaker}: expected "
            f"{TRAIN_PER_SPEAKER + TEST_PER_SPEAKER} embeddings, "
            f"found {len(embeddings)}"
        )

    train_embeddings = embeddings[:TRAIN_PER_SPEAKER]
    test_embeddings = embeddings[TRAIN_PER_SPEAKER:]

    train_names = names[:TRAIN_PER_SPEAKER]
    test_names = names[TRAIN_PER_SPEAKER:]

    return (
        train_names,
        train_embeddings,
        test_names,
        test_embeddings,
    )


def main() -> None:
    speakers = ["shiv", "friend"]

    print("=" * 70)
    print("LOADING ECAPA EMBEDDINGS")
    print("=" * 70)

    speaker_data = {}

    for speaker in speakers:
        names, embeddings = load_embeddings(speaker)

        speaker_data[speaker] = {
            "names": names,
            "embeddings": embeddings,
        }

        print(
            f"{speaker:10s}: "
            f"{embeddings.shape} embeddings"
        )

    # ---------------------------------------------------------
    # Train/test split
    # ---------------------------------------------------------

    train_embeddings = []
    train_labels = []

    test_embeddings = []
    test_labels = []
    test_names = []

    print()
    print("=" * 70)
    print("TRAIN / TEST SPLIT")
    print("=" * 70)

    for speaker in speakers:
        data = speaker_data[speaker]

        (
            train_names,
            speaker_train,
            speaker_test_names,
            speaker_test,
        ) = split_speaker_data(
            data["names"],
            data["embeddings"],
            speaker,
        )

        train_embeddings.append(speaker_train)
        train_labels.extend(
            [speaker] * len(speaker_train)
        )

        test_embeddings.append(speaker_test)
        test_labels.extend(
            [speaker] * len(speaker_test)
        )

        test_names.extend(
            [
                (speaker, name)
                for name in speaker_test_names
            ]
        )

        print(
            f"{speaker:10s}: "
            f"{len(speaker_train)} train / "
            f"{len(speaker_test)} test"
        )

    X_train = torch.cat(train_embeddings).numpy()
    X_test = torch.cat(test_embeddings).numpy()

    print()
    print(f"Training matrix : {X_train.shape}")
    print(f"Testing matrix  : {X_test.shape}")

    # ---------------------------------------------------------
    # Classifier
    # ---------------------------------------------------------

    classifier = Pipeline(
        [
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    print()
    print("=" * 70)
    print("TRAINING CLASSIFIER")
    print("=" * 70)

    classifier.fit(
        X_train,
        train_labels,
    )

    print("Classifier trained successfully.")

    # ---------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------

    predictions = classifier.predict(X_test)

    accuracy = accuracy_score(
        test_labels,
        predictions,
    )

    print()
    print("=" * 70)
    print("UNSEEN TEST RESULTS")
    print("=" * 70)

    for (speaker, name), expected, predicted in zip(
        test_names,
        test_labels,
        predictions,
    ):
        result = "✓" if expected == predicted else "✗"

        print()
        print(name)
        print(f"  Expected   : {expected}")
        print(f"  Prediction : {predicted}")
        print(f"  Result     : {result}")

    print()
    print("=" * 70)
    print("CLASSIFICATION RESULTS")
    print("=" * 70)

    print(
        f"\nAccuracy: {accuracy * 100:.2f}%"
    )

    print()
    print("Classification report:")
    print(
        classification_report(
            test_labels,
            predictions,
            labels=speakers,
            zero_division=0,
        )
    )

    print("Confusion matrix:")

    matrix = confusion_matrix(
        test_labels,
        predictions,
        labels=speakers,
    )

    print()
    print("             " + "  ".join(
        f"{speaker:>10s}"
        for speaker in speakers
    ))

    for speaker, row in zip(speakers, matrix):
        print(
            f"{speaker:>10s} "
            + "  ".join(
                f"{value:10d}"
                for value in row
            )
        )

    # ---------------------------------------------------------
    # Save classifier
    # ---------------------------------------------------------

    CLASSIFIER_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "classifier": classifier,
            "speakers": speakers,
            "embedding_dimension": 192,
        },
        CLASSIFIER_PATH,
    )

    print()
    print("=" * 70)
    print("MODEL SAVED")
    print("=" * 70)

    print(
        f"\nSaved classifier to: {CLASSIFIER_PATH}"
    )


if __name__ == "__main__":
    main()
