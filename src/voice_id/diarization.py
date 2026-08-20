import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier

from voice_id.segmentation import (
    load_vad,
    merge_segments,
    segment_audio,
)
from voice_id.unknowns import UnknownSpeakerManager


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"

DEVICE = (
    "cuda:0"
    if torch.cuda.is_available()
    else "cpu"
)

WINDOW_SECONDS = 2.0
STEP_SECONDS = 1.0

# Consecutive-window similarity below this
# indicates a possible speaker change.
TURN_THRESHOLD = 0.50

# Minimum similarity required to identify
# a known registered speaker.
KNOWN_THRESHOLD = 0.55

# Similarity required to match an existing
# unknown speaker.
UNKNOWN_MATCH_THRESHOLD = 0.55

# Minimum duration required before a segment
# can create a new unknown speaker.
MIN_UNKNOWN_SEGMENT_DURATION = 2.5

# Minimum known-speaker score required before
# refusing to create an unknown speaker.
NEW_UNKNOWN_MIN_KNOWN_SCORE = 0.55

# Temporal evidence threshold.
TEMPORAL_THRESHOLD = 0.45


# ============================================================
# MODEL
# ============================================================


def load_model() -> EncoderClassifier:
    """
    Load the pretrained ECAPA-TDNN speaker encoder.
    """

    return EncoderClassifier.from_hparams(
        source=MODEL_SOURCE,
        run_opts={
            "device": DEVICE,
        },
    )


# ============================================================
# AUDIO
# ============================================================


def load_audio(
    audio_path: Path,
) -> tuple[torch.Tensor, int]:
    """
    Load audio, convert to mono, and resample to 16 kHz.
    """

    signal, sample_rate = torchaudio.load(
        str(audio_path)
    )

    # Stereo -> mono
    if signal.shape[0] > 1:
        signal = signal.mean(
            dim=0,
            keepdim=True,
        )

    # ECAPA expects 16 kHz.
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate,
            new_freq=16000,
        )

        signal = resampler(signal)
        sample_rate = 16000

    return signal, sample_rate


# ============================================================
# EMBEDDINGS
# ============================================================


def extract_embedding(
    model: EncoderClassifier,
    audio: torch.Tensor,
) -> torch.Tensor:
    """
    Extract a normalized ECAPA speaker embedding.
    """

    audio = audio.to(DEVICE)

    with torch.no_grad():
        embedding = model.encode_batch(
            audio
        )

    embedding = embedding.squeeze()

    return F.normalize(
        embedding.unsqueeze(0),
        p=2,
        dim=1,
    ).squeeze(0)


def cosine_similarity(
    first: torch.Tensor,
    second: torch.Tensor,
) -> float:
    """
    Calculate cosine similarity between two embeddings.
    """

    first = F.normalize(
        first.unsqueeze(0),
        p=2,
        dim=1,
    ).squeeze(0)

    second = F.normalize(
        second.unsqueeze(0),
        p=2,
        dim=1,
    ).squeeze(0)

    return torch.dot(
        first,
        second,
    ).item()


# ============================================================
# KNOWN SPEAKER PROFILES
# ============================================================


def load_profiles() -> dict[str, torch.Tensor]:
    """
    Load all known-speaker embeddings and build
    one normalized mean profile per speaker.

    Expected structure:

        data/
        └── processed/
            └── known/
                ├── shiv/
                │   └── embeddings/
                │       └── *.pt
                └── friend/
                    └── embeddings/
                        └── *.pt
    """

    root = Path(
        "data/processed/known"
    )

    if not root.exists():
        return {}

    profiles: dict[
        str,
        torch.Tensor,
    ] = {}

    for speaker_dir in sorted(
        root.iterdir()
    ):

        if not speaker_dir.is_dir():
            continue

        embedding_dir = (
            speaker_dir / "embeddings"
        )

        if not embedding_dir.exists():
            continue

        embedding_files = sorted(
            embedding_dir.glob("*.pt")
        )

        if not embedding_files:
            continue

        embeddings = []

        for path in embedding_files:

            embedding = torch.load(
                path,
                map_location=DEVICE,
                weights_only=True,
            )

            embedding = F.normalize(
                embedding.unsqueeze(0),
                p=2,
                dim=1,
            ).squeeze(0)

            embeddings.append(
                embedding
            )

        matrix = torch.stack(
            embeddings
        )

        # Mean embedding becomes the
        # speaker profile.
        profile = matrix.mean(
            dim=0
        )

        profile = F.normalize(
            profile.unsqueeze(0),
            p=2,
            dim=1,
        ).squeeze(0)

        profiles[
            speaker_dir.name
        ] = profile



    return profiles


# ============================================================
# KNOWN SPEAKER MATCHING
# ============================================================


def get_known_scores(
    embedding: torch.Tensor,
    profiles: dict[str, torch.Tensor],
) -> dict[str, float]:

    return {
        speaker: cosine_similarity(
            embedding,
            profile,
        )
        for speaker, profile
        in profiles.items()
    }


def get_best_known_match(
    embedding: torch.Tensor,
    profiles: dict[str, torch.Tensor],
):
    scores = get_known_scores(
        embedding,
        profiles,
    )

    if not scores:
        return (
            None,
            None,
            {},
        )

    speaker = max(
        scores,
        key=scores.get,
    )

    return (
        speaker,
        scores[speaker],
        scores,
    )


# ============================================================
# UNKNOWN SPEAKER HELPERS
# ============================================================


def get_unknown_profile(
    speaker,
):
    """
    Safely retrieve an unknown-speaker profile.
    """

    if hasattr(
        speaker,
        "profile",
    ):
        return speaker.profile

    return None


def get_unknown_speakers(
    unknown_manager,
):
    """
    Return the registered unknown speakers.
    """

    if hasattr(
        unknown_manager,
        "get_speakers",
    ):
        return (
            unknown_manager.get_speakers()
        )

    return {}


def match_existing_unknown(
    embedding: torch.Tensor,
    unknown_manager,
):
    """
    Find the closest already-discovered
    unknown speaker.
    """

    speakers = get_unknown_speakers(
        unknown_manager
    )

    best_name = None
    best_score = -1.0

    for name, speaker in speakers.items():

        profile = get_unknown_profile(
            speaker
        )

        if profile is None:
            continue

        score = cosine_similarity(
            embedding,
            profile,
        )

        if score > best_score:
            best_score = score
            best_name = name

    return (
        best_name,
        best_score,
    )


# ============================================================
# WINDOW CREATION
# ============================================================


def create_windows(
    signal: torch.Tensor,
    sample_rate: int,
):
    """
    Create overlapping audio windows.
    """

    window_size = int(
        WINDOW_SECONDS
        * sample_rate
    )

    step_size = int(
        STEP_SECONDS
        * sample_rate
    )

    total_samples = (
        signal.shape[-1]
    )

    windows = []

    start = 0

    while (
        start + window_size
        <= total_samples
    ):

        end = (
            start
            + window_size
        )

        windows.append(
            (
                start / sample_rate,
                end / sample_rate,
                signal[
                    :,
                    start:end,
                ],
            )
        )

        start += step_size

    return windows


# ============================================================
# SPEAKER TURN DETECTION
# ============================================================


def detect_turn_points(
    model,
    signal,
    sample_rate,
    speech_segments,
):
    """
    Detect possible speaker changes using
    consecutive overlapping ECAPA embeddings.
    """

    windows = create_windows(
        signal,
        sample_rate,
    )

    if len(windows) < 2:
        return []

    embeddings = []

    for (
        start,
        end,
        audio,
    ) in windows:

        overlaps_speech = any(
            start < speech_end
            and end > speech_start
            for (
                speech_start,
                speech_end,
            ) in speech_segments
        )

        if not overlaps_speech:
            embeddings.append(
                None
            )
            continue

        embeddings.append(
            extract_embedding(
                model,
                audio,
            )
        )

    turn_points = []

    for index in range(
        len(embeddings) - 1
    ):

        first = embeddings[index]
        second = embeddings[
            index + 1
        ]

        if (
            first is None
            or second is None
        ):
            continue

        similarity = cosine_similarity(
            first,
            second,
        )

        if similarity < TURN_THRESHOLD:

            (
                first_start,
                first_end,
                _,
            ) = windows[index]

            (
                second_start,
                _,
                _,
            ) = windows[
                index + 1
            ]

            # Boundary between the two
            # overlapping windows.
            boundary = (
                first_end
                + second_start
            ) / 2.0

            turn_points.append(
                boundary
            )

    return turn_points


# ============================================================
# BUILD SEGMENTS
# ============================================================


def build_segments(
    speech_segments,
    turn_points,
):
    """
    Split speech regions at detected turn points.
    """

    segments = []

    for (
        speech_start,
        speech_end,
    ) in speech_segments:

        relevant_turns = [
            point
            for point in turn_points
            if (
                speech_start
                < point
                < speech_end
            )
        ]

        boundaries = [
            speech_start,
            *sorted(
                relevant_turns
            ),
            speech_end,
        ]

        for index in range(
            len(boundaries) - 1
        ):

            start = boundaries[
                index
            ]

            end = boundaries[
                index + 1
            ]

            if end <= start:
                continue

            segments.append(
                {
                    "start": start,
                    "end": end,
                    "duration": (
                        end - start
                    ),
                }
            )

    return segments


# ============================================================
# SEGMENT EMBEDDINGS
# ============================================================


def prepare_segments(
    model,
    signal,
    sample_rate,
    segments,
):
    """
    Extract an ECAPA embedding for every
    detected speaker segment.
    """

    prepared = []

    for segment in segments:

        start_sample = max(
            0,
            int(
                segment["start"]
                * sample_rate
            ),
        )

        end_sample = min(
            signal.shape[-1],
            int(
                segment["end"]
                * sample_rate
            ),
        )

        audio = signal[
            :,
            start_sample:end_sample,
        ]

        embedding = extract_embedding(
            model,
            audio,
        )

        prepared.append(
            {
                **segment,
                "embedding": embedding,
            }
        )

    return prepared


# ============================================================
# UNKNOWN IDENTITY CREATION
# ============================================================


def create_unknown(
    embedding,
    unknown_manager,
):
    """
    Register an observation with the unknown
    speaker manager.
    """

    result = unknown_manager.identify(
        embedding
    )

    name = result[0]
    score = result[1]
    created = result[2]

    return (
        name,
        score,
        created,
    )


# ============================================================
# INITIAL IDENTITY ASSIGNMENT
# ============================================================


def assign_initial_identities(
    segments,
    profiles,
    unknown_manager,
):
    """
    Assign known speakers, existing unknown speakers,
    new unknown speakers, or defer weak segments.

    No console output is produced here.
    """

    for segment in segments:

        embedding = segment[
            "embedding"
        ]

        duration = segment[
            "duration"
        ]

        (
            known_name,
            known_score,
            known_scores,
        ) = get_best_known_match(
            embedding,
            profiles,
        )

        segment[
            "known_scores"
        ] = known_scores

        segment[
            "known_best"
        ] = known_score

        # ----------------------------------------------------
        # 1. Known speaker
        # ----------------------------------------------------

        if (
            known_score is not None
            and known_score >= KNOWN_THRESHOLD
        ):

            segment[
                "speaker"
            ] = known_name

            segment[
                "confidence"
            ] = known_score

            segment[
                "source"
            ] = "known"

            continue

        # ----------------------------------------------------
        # 2. Existing unknown
        # ----------------------------------------------------

        (
            unknown_name,
            unknown_score,
        ) = match_existing_unknown(
            embedding,
            unknown_manager,
        )

        if (
            unknown_name is not None
            and unknown_score
            >= UNKNOWN_MATCH_THRESHOLD
        ):

            segment[
                "speaker"
            ] = unknown_name

            segment[
                "confidence"
            ] = unknown_score

            segment[
                "source"
            ] = "unknown_existing"

            # Register/update the observation.
            unknown_manager.identify(
                embedding
            )

            continue

        # ----------------------------------------------------
        # 3. New unknown
        # ----------------------------------------------------

        if (
            duration >= MIN_UNKNOWN_SEGMENT_DURATION
            and (
                known_score is None
                or known_score < NEW_UNKNOWN_MIN_KNOWN_SCORE
            )
        ):

            (
                new_name,
                new_score,
                created,
            ) = create_unknown(
                embedding,
                unknown_manager,
            )

            segment[
                "speaker"
            ] = new_name

            segment[
                "confidence"
            ] = new_score

            segment[
                "source"
            ] = (
                "unknown_new"
                if created
                else "unknown_existing"
            )

            continue

        # ----------------------------------------------------
        # 4. Weak / unresolved segment
        # ----------------------------------------------------

        segment[
            "speaker"
        ] = None

        segment[
            "confidence"
        ] = known_score

        segment[
            "source"
        ] = "deferred"

    return segments


# ============================================================
# TEMPORAL RESOLUTION
# ============================================================


def resolve_temporally(
    segments,
):
    """
    Attempt to resolve weak segments using
    neighboring identified segments.

    We do not automatically turn:

        shiv
        unknown
        shiv

    into:

        shiv
        shiv
        shiv

    unless the embedding itself provides enough
    temporal evidence.
    """

    for index, segment in enumerate(
        segments
    ):

        # Already identified.
        if (
            segment["speaker"]
            is not None
        ):
            continue

        embedding = segment[
            "embedding"
        ]

        candidates = []

        # ----------------------------------------------------
        # Previous identified segment
        # ----------------------------------------------------

        if index > 0:

            previous = segments[
                index - 1
            ]

            if (
                previous["speaker"]
                is not None
            ):

                score = cosine_similarity(
                    embedding,
                    previous[
                        "embedding"
                    ],
                )

                candidates.append(
                    (
                        score,
                        previous[
                            "speaker"
                        ],
                    )
                )

        # ----------------------------------------------------
        # Next identified segment
        # ----------------------------------------------------

        if (
            index + 1
            < len(segments)
        ):

            following = segments[
                index + 1
            ]

            if (
                following["speaker"]
                is not None
            ):

                score = cosine_similarity(
                    embedding,
                    following[
                        "embedding"
                    ],
                )

                candidates.append(
                    (
                        score,
                        following[
                            "speaker"
                        ],
                    )
                )

        if not candidates:
            continue

        best_score, best_name = max(
            candidates,
            key=lambda item: item[0],
        )

        if (
            best_score
            >= TEMPORAL_THRESHOLD
        ):

            segment[
                "speaker"
            ] = best_name

            segment[
                "confidence"
            ] = best_score

            segment[
                "source"
            ] = "temporal"

    return segments


# ============================================================
# FINAL TEMPORAL SMOOTHING
# ============================================================


def smooth_segments(
    segments,
):
    """
    Resolve an unresolved segment when both neighboring
    segments have the same already-known UNKNOWN identity.

    We deliberately do not force a known identity into
    an unresolved segment merely because it appears on
    both sides.
    """

    for index, segment in enumerate(
        segments
    ):

        if (
            segment["speaker"]
            is not None
        ):
            continue

        previous = (
            segments[index - 1]
            if index > 0
            else None
        )

        following = (
            segments[index + 1]
            if index + 1
            < len(segments)
            else None
        )

        if (
            previous is not None
            and following is not None
            and previous["speaker"]
            is not None
            and previous["speaker"]
            == following["speaker"]
            and previous["speaker"].startswith(
                "speaker_"
            )
        ):

            segment[
                "speaker"
            ] = previous[
                "speaker"
            ]

            segment[
                "confidence"
            ] = 0.0

            segment[
                "source"
            ] = "temporal_unknown"

    return segments


# ============================================================
# MERGE ADJACENT SAME-SPEAKER SEGMENTS
# ============================================================


def merge_identity_segments(
    segments,
):
    """
    Merge adjacent segments belonging to the
    same identity.
    """

    if not segments:
        return []

    merged = [
        {
            "start": segments[0][
                "start"
            ],
            "end": segments[0][
                "end"
            ],
            "speaker": segments[0][
                "speaker"
            ],
        }
    ]

    for current in segments[1:]:

        previous = merged[-1]

        if (
            previous["speaker"]
            == current["speaker"]
        ):

            previous[
                "end"
            ] = current["end"]

        else:

            merged.append(
                {
                    "start": current[
                        "start"
                    ],
                    "end": current[
                        "end"
                    ],
                    "speaker": current[
                        "speaker"
                    ],
                }
            )

    return merged


# ============================================================
# COMPLETE DIARIZATION API
# ============================================================


def diarize_audio(
    audio_path: Path,
) -> dict:
    """
    Complete reusable diarization API.

    This is the function that other applications,
    including Flask, should call.

    Returns actual values produced by the pipeline.
    """

    audio_path = Path(
        audio_path
    )

    if not audio_path.exists():
        raise FileNotFoundError(
            f"Audio file not found: "
            f"{audio_path}"
        )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = load_model()

    # --------------------------------------------------------
    # KNOWN PROFILES
    # --------------------------------------------------------

    profiles = load_profiles()

    # --------------------------------------------------------
    # AUDIO
    # --------------------------------------------------------

    signal, sample_rate = load_audio(
        audio_path
    )

    duration = (
        signal.shape[-1]
        / sample_rate
    )

    # --------------------------------------------------------
    # SPEECH SEGMENTATION
    # --------------------------------------------------------

    vad = load_vad()

    speech_segments = segment_audio(
        vad,
        audio_path,
    )

    speech_segments = merge_segments(
        speech_segments
    )

    if not speech_segments:

        return {
            "audio": {
                "path": str(
                    audio_path
                ),
                "sample_rate": (
                    sample_rate
                ),
                "duration": (
                    duration
                ),
            },
            "known_speakers": sorted(
                profiles.keys()
            ),
            "speech_segments": [],
            "turn_points": [],
            "segments": [],
            "unknown_speakers": [],
        }

    # --------------------------------------------------------
    # SPEAKER TURN DETECTION
    # --------------------------------------------------------

    turn_points = detect_turn_points(
        model,
        signal,
        sample_rate,
        speech_segments,
    )

    # --------------------------------------------------------
    # BUILD SEGMENTS
    # --------------------------------------------------------

    segments = build_segments(
        speech_segments,
        turn_points,
    )

    # --------------------------------------------------------
    # SEGMENT EMBEDDINGS
    # --------------------------------------------------------

    segments = prepare_segments(
        model,
        signal,
        sample_rate,
        segments,
    )

    # --------------------------------------------------------
    # UNKNOWN MANAGER
    # --------------------------------------------------------

    unknown_manager = UnknownSpeakerManager(
        match_threshold=(
            UNKNOWN_MATCH_THRESHOLD
        )
    )

    # --------------------------------------------------------
    # INITIAL IDENTITY ASSIGNMENT
    # --------------------------------------------------------

    segments = assign_initial_identities(
        segments,
        profiles,
        unknown_manager,
    )

    # --------------------------------------------------------
    # TEMPORAL RESOLUTION
    # --------------------------------------------------------

    segments = resolve_temporally(
        segments
    )

    # --------------------------------------------------------
    # TEMPORAL SMOOTHING
    # --------------------------------------------------------

    segments = smooth_segments(
        segments
    )

    # --------------------------------------------------------
    # FINAL MERGE
    # --------------------------------------------------------

    final_segments = (
        merge_identity_segments(
            segments
        )
    )

    # --------------------------------------------------------
    # UNKNOWN SPEAKERS
    # --------------------------------------------------------

    unknown_speakers = []

    speakers = get_unknown_speakers(
        unknown_manager
    )

    for name, speaker in (
        speakers.items()
    ):

        observations = getattr(
            speaker,
            "observations",
            None,
        )

        unknown_speakers.append(
            {
                "id": name,
                "observations": (
                    observations
                ),
            }
        )

    # --------------------------------------------------------
    # SERIALIZABLE FINAL SEGMENTS
    # --------------------------------------------------------

    serializable_segments = []

    for segment in final_segments:

        start = float(
            segment["start"]
        )

        end = float(
            segment["end"]
        )

        serializable_segments.append(
            {
                "start": start,
                "end": end,
                "duration": end - start,
                "speaker": segment[
                    "speaker"
                ],
            }
        )

    # --------------------------------------------------------
    # RETURN STRUCTURED RESULT
    # --------------------------------------------------------

    return {
        "audio": {
            "path": str(
                audio_path
            ),
            "sample_rate": sample_rate,
            "duration": duration,
        },
        "known_speakers": sorted(
            profiles.keys()
        ),
        "speech_segments": [
            {
                "start": float(start),
                "end": float(end),
                "duration": float(
                    end - start
                ),
            }
            for start, end
            in speech_segments
        ],
        "turn_points": [
            float(point)
            for point in turn_points
        ],
        "segments": (
            serializable_segments
        ),
        "unknown_speakers": (
            unknown_speakers
        ),
    }


# ============================================================
# CLI
# ============================================================


def print_result(
    result: dict,
) -> None:
    """
    Human-readable CLI presentation.

    This is intentionally separate from the actual
    diarization engine.
    """

    print(
        "=" * 70
    )
    print(
        "VOICE DIARIZATION"
    )
    print(
        "=" * 70
    )

    audio = result["audio"]

    print()
    print(
        "AUDIO"
    )
    print(
        "-" * 70
    )

    print(
        f"File        : "
        f"{audio['path']}"
    )

    print(
        f"Sample rate : "
        f"{audio['sample_rate']} Hz"
    )

    print(
        f"Duration    : "
        f"{audio['duration']:.2f}s"
    )

    print()
    print(
        "KNOWN SPEAKERS"
    )
    print(
        "-" * 70
    )

    for speaker in result[
        "known_speakers"
    ]:
        print(
            f"  {speaker}"
        )

    print()
    print(
        "SPEECH SEGMENTS"
    )
    print(
        "-" * 70
    )

    for index, segment in enumerate(
        result["speech_segments"],
        start=1,
    ):

        print(
            f"[{index:02d}] "
            f"{segment['start']:.2f}s → "
            f"{segment['end']:.2f}s "
            f"({segment['duration']:.2f}s)"
        )

    print()
    print(
        "SPEAKER TURN POINTS"
    )
    print(
        "-" * 70
    )

    if result["turn_points"]:

        for point in result[
            "turn_points"
        ]:
            print(
                f"  → {point:.2f}s"
            )

    else:
        print(
            "  None detected"
        )

    print()
    print(
        "FINAL DIARIZATION"
    )
    print(
        "-" * 70
    )

    for index, segment in enumerate(
        result["segments"],
        start=1,
    ):

        print()
        print(
            f"[{index}] "
            f"{segment['start']:.2f}s → "
            f"{segment['end']:.2f}s"
        )

        print(
            f"    Speaker : "
            f"{segment['speaker']}"
        )

        print(
            f"    Duration: "
            f"{segment['duration']:.2f}s"
        )

    print()
    print(
        "DISCOVERED UNKNOWN SPEAKERS"
    )
    print(
        "-" * 70
    )

    if result[
        "unknown_speakers"
    ]:

        for speaker in result[
            "unknown_speakers"
        ]:

            print(
                f"{speaker['id']}: "
                f"{speaker['observations']} "
                f"observations"
            )

    else:
        print(
            "None"
        )

    print()
    print(
        "=" * 70
    )
    print(
        "DONE"
    )
    print(
        "=" * 70
    )


def main() -> None:

    if len(sys.argv) != 2:

        print(
            "Usage:"
        )

        print(
            "  uv run python -m "
            "voice_id.diarization "
            "<audio.wav>"
        )

        raise SystemExit(1)

    audio_path = Path(
        sys.argv[1]
    )

    result = diarize_audio(
        audio_path
    )

    print_result(
        result
    )


if __name__ == "__main__":
    main()
