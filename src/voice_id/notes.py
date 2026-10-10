"""Notes generation stage.

Turns a pipeline result (speaker-attributed transcript) into structured,
timestamped notes using Gemini Flash, then validates everything Gemini
returns against the real transcript before it reaches the UI.

Pure module: no Flask, no web imports.

    uv run python -m voice_id.notes result.json
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

from pydantic import BaseModel


class NotesError(Exception):
    """Raised when notes cannot be generated from the given result."""


class NotesContentError(NotesError):
    """The transcript itself has too little content for notes."""


# ---------------------------------------------------------------------------
# Schema — what we FORCE Gemini to return (structured output, no markdown)
# ---------------------------------------------------------------------------

class Topic(BaseModel):
    name: str
    summary: str
    start: float
    end: float
    key_points: list[str] = []


class Concept(BaseModel):
    term: str
    definition: str
    timestamps: list[float] = []
    search_query: str = ""


class SpeakerContribution(BaseModel):
    speaker: str
    points: list[str] = []


class ActionItem(BaseModel):
    task: str
    owner: str | None = None
    timestamp: float | None = None


class Notes(BaseModel):
    title: str
    tldr: str
    mode: str
    topics: list[Topic] = []
    concepts: list[Concept] = []
    speaker_contributions: list[SpeakerContribution] = []
    action_items: list[ActionItem] = []
    diagram: str = ""


# ---------------------------------------------------------------------------
# Result adapter — THE ONLY CODE THAT TOUCHES YOUR RESULT'S SHAPE.
#
# Written from your README + UI screenshot (rows look like
# {start, end, speaker, text}). Each lookup tries a few likely key names;
# if none match it fails loudly and says what to fix.
# ---------------------------------------------------------------------------

@dataclass
class Segment:
    start: float
    end: float
    speaker: str
    text: str


def _get(d, *keys, default=None):
    if not isinstance(d, dict):
        return default
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def _unwrap(result):
    """Accept the full API response {success, result} OR the inner result."""
    if isinstance(result, dict) and isinstance(result.get("result"), dict):
        return result["result"]
    return result


def extract_segments(result) -> list[Segment]:
    raw = _get(result, "transcript", "segments", "aligned", default=[])
    segments = []
    for i, item in enumerate(raw):
        start = _get(item, "start", "start_time", "start_s")
        end = _get(item, "end", "end_time", "end_s")
        text = _get(item, "text", "content", "transcript", default="")
        speaker = _get(item, "speaker", "speaker_id", "speaker_label", default="unknown")
        if start is None or end is None:
            raise NotesError(
                f"transcript[{i}] has no recognizable time field — "
                "paste your /api/analyze response so we can fix the adapter."
            )
        segments.append(Segment(float(start), float(end), str(speaker), str(text)))
    if not segments:
        raise NotesError("No transcript segments found in the result.")
    return segments


def extract_speakers(result) -> list[str]:
    raw = _get(result, "speakers", "detected_speakers", default=[])
    names = []
    for s in raw:
        if isinstance(s, dict):
            s = _get(s, "id", "name", "speaker")
        if s:
            names.append(str(s))
    return names


def extract_duration(result, segments) -> float:
    audio = _get(result, "audio", default={})
    d = _get(audio, "duration", "duration_s", "length") if isinstance(audio, dict) else None
    return float(d) if d else max(s.end for s in segments)


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

NOTES_PROMPT = """Below is a timestamped, speaker-attributed transcript of an audio recording,
produced by automatic speech recognition and speaker diarization.

Generate structured notes from it as JSON. Follow these rules:

1. Use only what the transcript says. Do not invent facts.
2. Every timestamp you output MUST be copied from the transcript:
   - topic "start" = exact start time of the segment where the topic begins
   - topic "end"   = exact end time of the segment where it finishes
   - concept/action timestamps = exact start time of a relevant segment
3. The transcript may contain mis-heard words from speech recognition.
   Silently correct obvious errors in your own output text.
4. "mode" must be one of: lecture, meeting, interview, conversation, podcast.
5. "tldr": 2-4 sentences summarizing the whole recording.
6. "topics": main things discussed, in chronological order, with short
   "key_points" bullets. Most recordings have 2-6 topics.
7. "concepts": important terms, names or ideas mentioned. Each needs a
   1-2 sentence "definition" and a clean "search_query" someone could paste
   into a search engine (corrected spelling, no speaker names).
8. "speaker_contributions": per speaker (use their EXACT labels from the
   transcript), 1-4 bullets of what they contributed. Skip speakers who
   only said greetings or fillers.
9. "action_items": tasks, plans or decisions that came up. "owner" is the
   responsible speaker's label or null. Empty list if none.
10. "diagram": a Mermaid "graph TD" flowchart (max 12 nodes) showing the
    flow of the discussion. Short node labels. Empty string if it does not
    suit the content.
11. Respect the caps: max 8 topics, 12 concepts, 6 action items,
    5 speaker contributions.
12. If the transcript is trivial, return empty lists and a short tldr.

TRANSCRIPT:
"""


def build_prompt(segments, speakers, duration) -> str:
    lines = [f"Audio duration: {duration:.1f} seconds."]
    if speakers:
        lines.append("Detected speakers: " + ", ".join(speakers))
    lines.append("")
    for s in segments:
        lines.append(f"[{s.start:.1f} - {s.end:.1f}] {s.speaker}: {s.text}")
    return NOTES_PROMPT + "\n".join(lines)


# ---------------------------------------------------------------------------
# Gemini call
# ---------------------------------------------------------------------------

def _call_gemini(prompt: str) -> Notes:
    from google import genai
    from google.genai import types

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise NotesError("GEMINI_API_KEY is not set (add it to .env).")

    # .env NOTES_MODEL overrides this default
    model = os.environ.get("NOTES_MODEL", "gemini-3.5-flash")
    client = genai.Client(api_key=api_key)
    config = types.GenerateContentConfig(
        temperature=0.3,
        response_mime_type="application/json",
        response_schema=Notes,
    )

    last_error = None
    for attempt in range(3):  # free-tier 429s / blips -> simple backoff
        try:
            # Chat-based pattern (recommended by the SDK; silences the
            # AFC warning that Models.generate_content triggers)
            chat = client.chats.create(model=model, config=config)
            response = chat.send_message(prompt)

            if response.parsed is not None:
                return response.parsed
            return Notes.model_validate_json(response.text)
        except Exception as error:
            last_error = error
            time.sleep(2 * (attempt + 1))
    raise NotesError(f"Gemini notes request failed after retries: {last_error}")


# ---------------------------------------------------------------------------
# Validation — trust nothing Gemini said until it matches the real transcript
# ---------------------------------------------------------------------------

ALLOWED_MODES = {"lecture", "meeting", "interview", "conversation", "podcast"}
MERMAID_STARTS = ("graph", "flowchart", "sequencediagram", "classdiagram",
                  "statediagram", "mindmap", "timeline", "gantt", "pie", "erdiagram")

TOPIC_CAP, CONCEPT_CAP, ACTION_CAP, CONTRIBUTION_CAP = 8, 12, 6, 5
SNAP_TOLERANCE = 5.0  # seconds — how far a hallucinated timestamp may drift


def _snap(timestamp, starts, tolerance=SNAP_TOLERANCE):
    """Nearest REAL segment start within tolerance, else None."""
    if timestamp is None:
        return None
    best = min(starts, key=lambda s: abs(s - float(timestamp)))
    return float(best) if abs(best - float(timestamp)) <= tolerance else None


def _clean_notes(notes: Notes, segments, speakers, duration) -> dict:
    starts = [s.start for s in segments]
    ends_by_start = {s.start: s.end for s in segments}
    speaker_lookup = {name.lower(): name for name in speakers}

    # topics — start must snap to a real segment, else the topic is dropped
    topics = []
    for topic in notes.topics[:TOPIC_CAP]:
        start = _snap(topic.start, starts)
        if start is None:
            continue
        end = min(max(topic.end, start), duration)
        end = max(end, ends_by_start.get(start, start))  # at least cover its opening segment
        topics.append({
            "name": topic.name.strip(),
            "summary": topic.summary.strip(),
            "start": start,
            "end": end,
            "key_points": [p.strip() for p in topic.key_points if p.strip()],
        })

    # concepts — timestamps snapped, deduped, sorted
    concepts = []
    for concept in notes.concepts[:CONCEPT_CAP]:
        term = concept.term.strip()
        if not term:
            continue
        stamps = []
        for ts in concept.timestamps[:5]:
            snapped = _snap(ts, starts)
            if snapped is not None and snapped not in stamps:
                stamps.append(snapped)
        stamps.sort()
        concepts.append({
            "term": term,
            "definition": concept.definition.strip(),
            "timestamps": stamps,
            "search_query": concept.search_query.strip(),
        })

    # speaker contributions — unknown/hallucinated speakers dropped
    contributions = []
    for contribution in notes.speaker_contributions[:CONTRIBUTION_CAP]:
        speaker = speaker_lookup.get(contribution.speaker.lower().strip())
        points = [p.strip() for p in contribution.points if p.strip()]
        if speaker and points:
            contributions.append({"speaker": speaker, "points": points})

    # action items
    action_items = []
    for item in notes.action_items[:ACTION_CAP]:
        task = item.task.strip()
        if not task:
            continue
        action_items.append({
            "task": task,
            "owner": (item.owner or "").strip() or None,
            "timestamp": _snap(item.timestamp, starts),
        })

    # scalars
    mode = notes.mode.lower().strip()
    if mode not in ALLOWED_MODES:
        mode = "conversation"

    diagram = notes.diagram.strip()
    if not diagram.lower().startswith(MERMAID_STARTS) or len(diagram) > 4000:
        diagram = ""  # client hides empty diagrams gracefully

    title = notes.title.strip() or "Session notes"
    tldr = notes.tldr.strip()

    if not (topics or concepts or tldr):
        raise NotesContentError("Not enough meaningful content in the transcript for notes.")

    return {
        "notes_version": 1,
        "title": title,
        "tldr": tldr,
        "mode": mode,
        "topics": topics,
        "concepts": concepts,
        "speaker_contributions": contributions,
        "action_items": action_items,
        "diagram": diagram,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def generate_notes(result) -> dict:
    """Pipeline result -> validated notes dict."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    result = _unwrap(result)
    segments = extract_segments(result)
    speakers = extract_speakers(result)
    duration = extract_duration(result, segments)

    if sum(len(s.text.split()) for s in segments) < 20:
        raise NotesContentError("Transcript is too short to generate notes from.")

    prompt = build_prompt(segments, speakers, duration)
    raw_notes = _call_gemini(prompt)
    return _clean_notes(raw_notes, segments, speakers, duration)


if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) != 2:
        raise SystemExit("Usage: uv run python -m voice_id.notes <result.json>")
    with open(sys.argv[1], encoding="utf-8") as handle:
        payload = json.load(handle)
    print(json.dumps(generate_notes(payload), indent=2, ensure_ascii=False))
