from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class UnknownSpeaker:
    """Represents an unknown speaker discovered during a session."""

    name: str
    profile: torch.Tensor
    observations: int = 1


class UnknownSpeakerManager:
    """
    Manage dynamically discovered unknown speakers.

    Known speakers such as 'shiv' and 'friend' are handled
    elsewhere. This class only receives embeddings that have
    already been rejected by the known-speaker system.

    Example:

        embedding
            ↓
        UnknownSpeakerManager
            ↓
        existing speaker_02?
          /          \
        yes           no
        ↓              ↓
    speaker_02     create speaker_03
    """

    def __init__(
        self,
        match_threshold: float = 0.55,
    ) -> None:

        self.match_threshold = match_threshold

        self.speakers: dict[
            str,
            UnknownSpeaker,
        ] = {}

        self.next_id = 2

    # =====================================================
    # NORMALIZATION
    # =====================================================

    @staticmethod
    def normalize(
        embedding: torch.Tensor,
    ) -> torch.Tensor:

        return F.normalize(
            embedding.unsqueeze(0),
            p=2,
            dim=1,
        ).squeeze(0)

    # =====================================================
    # SIMILARITY
    # =====================================================

    @staticmethod
    def similarity(
        embedding_a: torch.Tensor,
        embedding_b: torch.Tensor,
    ) -> float:

        embedding_a = UnknownSpeakerManager.normalize(
            embedding_a
        )

        embedding_b = UnknownSpeakerManager.normalize(
            embedding_b
        )

        return torch.dot(
            embedding_a,
            embedding_b,
        ).item()

    # =====================================================
    # FIND MATCH
    # =====================================================

    def find_match(
        self,
        embedding: torch.Tensor,
    ) -> tuple[str | None, float]:

        if not self.speakers:
            return None, -1.0

        embedding = self.normalize(
            embedding
        )

        best_name = None
        best_score = -1.0

        for name, speaker in self.speakers.items():

            score = torch.dot(
                embedding,
                speaker.profile,
            ).item()

            if score > best_score:

                best_score = score
                best_name = name

        if (
            best_score
            >= self.match_threshold
        ):

            return (
                best_name,
                best_score,
            )

        return (
            None,
            best_score,
        )

    # =====================================================
    # CREATE SPEAKER
    # =====================================================

    def create_speaker(
        self,
        embedding: torch.Tensor,
    ) -> str:

        name = f"speaker_{self.next_id:02d}"

        embedding = self.normalize(
            embedding
        )

        self.speakers[name] = UnknownSpeaker(
            name=name,
            profile=embedding.clone(),
            observations=1,
        )

        self.next_id += 1

        return name

    # =====================================================
    # UPDATE PROFILE
    # =====================================================

    def update_profile(
        self,
        name: str,
        embedding: torch.Tensor,
    ) -> None:

        if name not in self.speakers:
            raise KeyError(
                f"Unknown speaker does not exist: "
                f"{name}"
            )

        speaker = self.speakers[name]

        embedding = self.normalize(
            embedding
        )

        # Running mean.
        count = speaker.observations

        updated_profile = (
            speaker.profile * count
            + embedding
        ) / (count + 1)

        speaker.profile = self.normalize(
            updated_profile
        )

        speaker.observations += 1

    # =====================================================
    # REGISTER / MATCH
    # =====================================================

    def identify(
        self,
        embedding: torch.Tensor,
    ) -> tuple[str, float, bool]:

        match_name, match_score = (
            self.find_match(
                embedding
            )
        )

        # Existing unknown speaker.
        if match_name is not None:

            self.update_profile(
                match_name,
                embedding,
            )

            return (
                match_name,
                match_score,
                False,
            )

        # Completely new unknown speaker.
        new_name = self.create_speaker(
            embedding
        )

        return (
            new_name,
            1.0,
            True,
        )

    # =====================================================
    # INFORMATION
    # =====================================================

    def get_speakers(
        self,
    ) -> dict[str, UnknownSpeaker]:

        return self.speakers.copy()

    def __len__(self) -> int:

        return len(self.speakers)



if __name__ == "__main__":

    print("=" * 70)
    print("UNKNOWN SPEAKER MANAGER TEST")
    print("=" * 70)

    manager = UnknownSpeakerManager(
        match_threshold=0.55
    )

    torch.manual_seed(42)

    # Simulated first unknown speaker.
    speaker_a = F.normalize(
        torch.randn(192),
        dim=0,
    )

    # Similar to speaker_a.
    speaker_a_2 = F.normalize(
        speaker_a
        + 0.10 * torch.randn(192),
        dim=0,
    )

    # Completely different speaker.
    speaker_b = F.normalize(
        torch.randn(192),
        dim=0,
    )

    print("\nFirst unknown:")
    name, score, created = manager.identify(
        speaker_a
    )

    print(f"  Identity : {name}")
    print(f"  Score    : {score:.4f}")
    print(f"  Created  : {created}")

    print("\nSecond observation:")
    name, score, created = manager.identify(
        speaker_a_2
    )

    print(f"  Identity : {name}")
    print(f"  Score    : {score:.4f}")
    print(f"  Created  : {created}")

    print("\nDifferent unknown:")
    name, score, created = manager.identify(
        speaker_b
    )

    print(f"  Identity : {name}")
    print(f"  Score    : {score:.4f}")
    print(f"  Created  : {created}")

    print("\nRegistered speakers:")

    for name, speaker in manager.get_speakers().items():

        print(
            f"  {name}: "
            f"{speaker.observations} observations"
        )
