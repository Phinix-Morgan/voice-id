from pathlib import Path

import matplotlib.pyplot as plt
import torch


EMBEDDINGS_DIR = Path("data/processed/shiv/embeddings")


def load_embeddings() -> tuple[list[str], torch.Tensor]:
    """Load all speaker embeddings onto the GPU."""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    embedding_files = sorted(EMBEDDINGS_DIR.glob("*.pt"))

    if not embedding_files:
        raise FileNotFoundError(
            f"No embeddings found in {EMBEDDINGS_DIR}"
        )

    names = []
    embeddings = []

    for path in embedding_files:
        embedding = torch.load(
            path,
            map_location=device,
            weights_only=True,
        )

        names.append(path.stem)
        embeddings.append(embedding)

    return names, torch.stack(embeddings)


def pca_2d(embeddings: torch.Tensor) -> torch.Tensor:
    """Project embeddings from 192D to 2D using PCA."""
    # Center the data.
    centered = embeddings - embeddings.mean(dim=0, keepdim=True)

    # SVD is performed on the GPU because centered is on cuda:0.
    _, _, V = torch.pca_lowrank(
        centered,
        q=2,
        center=False,
    )

    # Project onto the first two principal components.
    return centered @ V[:, :2]


def main() -> None:
    names, embeddings = load_embeddings()

    print(f"Embeddings: {embeddings.shape}")
    print(f"Device:     {embeddings.device}")

    projected = pca_2d(embeddings)

    print(f"PCA result: {projected.shape}")
    print(f"Device:     {projected.device}")

    # Only move the small 15 x 2 result to CPU for Matplotlib.
    points = projected.cpu().numpy()

    plt.figure(figsize=(10, 8))

    plt.scatter(
        points[:, 0],
        points[:, 1],
        s=100,
    )

    for index, name in enumerate(names):
        plt.annotate(
            f"S{index + 1:02d}",
            (points[index, 0], points[index, 1]),
            xytext=(6, 6),
            textcoords="offset points",
        )

    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("ECAPA-TDNN Speaker Embeddings — Shiv")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
