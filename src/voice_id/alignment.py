from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class DiarizationSegment:
    start: float
    end: float
    speaker: str | None


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str


@dataclass
class AlignedSegment:
    start: float
    end: float
    speaker: str
    text: str


def overlap_duration(
    start_a: float,
    end_a: float,
    start_b: float,
    end_b: float,
) -> float:
    start = max(start_a, start_b)
    end = min(end_a, end_b)

    return max(0.0, end - start)


def find_best_speaker(
    transcript_segment: TranscriptSegment,
    diarization_segments: list[DiarizationSegment],
) -> tuple[str, float]:
    best_speaker = "unknown"
    best_overlap = 0.0

    for diarization_segment in diarization_segments:
        if diarization_segment.speaker is None:
            continue

        overlap = overlap_duration(
            transcript_segment.start,
            transcript_segment.end,
            diarization_segment.start,
            diarization_segment.end,
        )

        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = diarization_segment.speaker

    return best_speaker, best_overlap


def align_transcript(
    diarization_segments: list[DiarizationSegment],
    transcript_segments: list[TranscriptSegment],
) -> list[AlignedSegment]:
    aligned: list[AlignedSegment] = []

    for transcript_segment in transcript_segments:
        speaker, _ = find_best_speaker(
            transcript_segment,
            diarization_segments,
        )

        aligned.append(
            AlignedSegment(
                start=transcript_segment.start,
                end=transcript_segment.end,
                speaker=speaker,
                text=transcript_segment.text,
            )
        )

    return aligned


def merge_adjacent_segments(
    segments: list[AlignedSegment],
) -> list[AlignedSegment]:
    if not segments:
        return []

    merged: list[AlignedSegment] = [segments[0]]

    for current in segments[1:]:
        previous = merged[-1]

        if (
            previous.speaker == current.speaker
            and current.start <= previous.end + 0.5
        ):
            previous.end = max(
                previous.end,
                current.end,
            )

            previous.text = (
                previous.text.rstrip()
                + " "
                + current.text.lstrip()
            )

        else:
            merged.append(current)

    return merged


def align(
    diarization_data: list[dict[str, Any]],
    transcript_data: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    diarization_segments = [
        DiarizationSegment(
            start=float(segment["start"]),
            end=float(segment["end"]),
            speaker=segment.get("speaker"),
        )
        for segment in diarization_data
    ]

    transcript_segments = [
        TranscriptSegment(
            start=float(segment["start"]),
            end=float(segment["end"]),
            text=str(segment["text"]).strip(),
        )
        for segment in transcript_data
        if str(segment.get("text", "")).strip()
    ]

    aligned = align_transcript(
        diarization_segments,
        transcript_segments,
    )

    aligned = merge_adjacent_segments(aligned)

    return [
        {
            "start": segment.start,
            "end": segment.end,
            "speaker": segment.speaker,
            "text": segment.text,
        }
        for segment in aligned
    ]


def print_aligned_transcript(
    segments: list[dict[str, Any]],
) -> None:
    print()
    print("=" * 70)
    print("SPEAKER-ATTRIBUTED TRANSCRIPT")
    print("=" * 70)

    if not segments:
        print("No aligned transcript segments.")
        return

    for segment in segments:
        print(
            f"[{segment['start']:07.2f}s → "
            f"{segment['end']:07.2f}s]"
        )
        print(f"{segment['speaker'].upper()}")
        print(segment["text"])
        print()


def main() -> None:
    """
    Small standalone test.

    This does not run the ML pipeline.
    It only verifies the alignment algorithm.
    """

    diarization = [
        {
            "start": 0.01,
            "end": 5.50,
            "speaker": "shiv",
        },
        {
            "start": 5.50,
            "end": 10.50,
            "speaker": "speaker_02",
        },
        {
            "start": 10.50,
            "end": 15.50,
            "speaker": "shiv",
        },
    ]

    transcript = [
        {
            "start": 0.00,
            "end": 5.00,
            "text": "Hey, Sam. Are you free to grab coffee right now?",
        },
        {
            "start": 5.00,
            "end": 9.00,
            "text": "I wish I could. I'm stuck at my desk working on a tight deadline.",
        },
        {
            "start": 9.00,
            "end": 13.00,
            "text": "Oh, no. But can you take a quick 10-minute break?",
        },
    ]

    result = align(
        diarization,
        transcript,
    )

    print_aligned_transcript(result)


if __name__ == "__main__":
    main()
