from pathlib import Path

import torch
import torch.nn.functional as F


KNOWN_DIR = Path("data/processed/known")


def load_speaker_embeddings(
    speaker: str,
) -> tuple[list[str], torch.Tensor]:
    """Load all embeddings for a known speaker."""

    embeddings_dir = KNOWN_DIR / speaker / "embeddings"

    if not embeddings_dir.exists():
        raise FileNotFoundError(
            f"Embeddings directory not found: {embeddings_dir}"
        )

    embedding_files = sorted(embeddings_dir.glob("*.pt"))

    if not embedding_files:
        raise FileNotFoundError(
            f"No embeddings found in {embeddings_dir}"
        )

    names = []
    embeddings = []

    for path in embedding_files:
        embedding = torch.load(
            path,
            map_location="cuda:0" if torch.cuda.is_available() else "cpu",
            weights_only=True,
        )

        names.append(path.stem)
        embeddings.append(embedding)

    matrix = torch.stack(embeddings)

    return names, matrix


def pairwise_similarities(
    embeddings_a: torch.Tensor,
    embeddings_b: torch.Tensor,
) -> torch.Tensor:
    """Calculate cosine similarity between two embedding sets."""

    normalized_a = F.normalize(embeddings_a, p=2, dim=1)
    normalized_b = F.normalize(embeddings_b, p=2, dim=1)

    return normalized_a @ normalized_b.T


def analyze_same_speaker(
    similarities: torch.Tensor,
) -> torch.Tensor:
    """Return unique same-speaker comparisons."""

    n = similarities.shape[0]

    indices = torch.triu_indices(
        n,
        n,
        offset=1,
        device=similarities.device,
    )

    return similarities[indices[0], indices[1]]


def print_statistics(
    title: str,
    similarities: torch.Tensor,
) -> None:
    """Print similarity statistics."""

    print("=" * 70)
    print(title)
    print("=" * 70)

    print(f"Comparisons : {len(similarities)}")
    print(f"Mean        : {similarities.mean().item():.4f}")
    print(
        f"Median      : {similarities.median().item():.4f}"
    )
    print(
        f"Std         : {similarities.std().item():.4f}"
    )
    print(
        f"Minimum     : {similarities.min().item():.4f}"
    )
    print(
        f"Maximum     : {similarities.max().item():.4f}"
    )


def main() -> None:
    device = (
        "cuda:0"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Loading speaker embeddings...")

    shiv_names, shiv_embeddings = load_speaker_embeddings(
        "shiv"
    )

    friend_names, friend_embeddings = load_speaker_embeddings(
        "friend"
    )

    shiv_embeddings = shiv_embeddings.to(device)
    friend_embeddings = friend_embeddings.to(device)

    print()
    print(
        f"Shiv embeddings       : {shiv_embeddings.shape}"
    )
    print(
        f"Friend embeddings     : {friend_embeddings.shape}"
    )
    print(f"Device                : {device}")

    # ---------------------------------------------------------
    # Same-speaker comparisons
    # ---------------------------------------------------------

    shiv_same_matrix = pairwise_similarities(
        shiv_embeddings,
        shiv_embeddings,
    )

    friend_same_matrix = pairwise_similarities(
        friend_embeddings,
        friend_embeddings,
    )

    shiv_same = analyze_same_speaker(
        shiv_same_matrix
    )

    friend_same = analyze_same_speaker(
        friend_same_matrix
    )

    # ---------------------------------------------------------
    # Different-speaker comparisons
    # ---------------------------------------------------------

    cross_matrix = pairwise_similarities(
        shiv_embeddings,
        friend_embeddings,
    )

    cross = cross_matrix.flatten()

    # ---------------------------------------------------------
    # Results
    # ---------------------------------------------------------

    print()

    print_statistics(
        "SHIV ↔ SHIV",
        shiv_same,
    )

    print()

    print_statistics(
        "FRIEND ↔ FRIEND",
        friend_same,
    )

    print()

    print_statistics(
        "SHIV ↔ FRIEND",
        cross,
    )

    # ---------------------------------------------------------
    # Overall separation
    # ---------------------------------------------------------

    same_all = torch.cat(
        [shiv_same, friend_same]
    )

    same_min = same_all.min().item()
    same_max = same_all.max().item()

    different_min = cross.min().item()
    different_max = cross.max().item()

    print()
    print("=" * 70)
    print("OVERALL COMPARISON")
    print("=" * 70)

    print(
        f"Same-speaker range      : "
        f"{same_min:.4f} → {same_max:.4f}"
    )

    print(
        f"Different-speaker range : "
        f"{different_min:.4f} → {different_max:.4f}"
    )

    print(
        f"\nHighest different-speaker score : "
        f"{different_max:.4f}"
    )

    print(
        f"Lowest same-speaker score        : "
        f"{same_min:.4f}"
    )

    overlap = (
        different_max >= same_min
    )

    print(
        "\nDistribution overlap detected : "
        f"{'YES' if overlap else 'NO'}"
    )


if __name__ == "__main__":
    main()
