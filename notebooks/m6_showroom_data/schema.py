"""Shared showroom annotation contract. IDs refer to records, never file positions."""

from typing import NotRequired, TypedDict

INTENTS = ("UNK", "greet", "affirm", "deny", "inform", "request_flow",
           "request_stop", "request_resume", "request_quit")
SLOTS = ("name", "flow_ref", "part", "theme", "choice")
SLOT_LABELS = ("PAD", "UNK", "O") + tuple(f"{prefix}-{slot}" for slot in SLOTS for prefix in ("B", "I"))
SPLITS = ("train", "dev", "test")
RATIOS = (0.8, 0.1, 0.1)
ALLOWED = {intent: set() for intent in INTENTS}
ALLOWED.update(greet={"name"}, inform=set(SLOTS), request_flow={"flow_ref", "part", "theme"})
VALUES = {"flow_ref": {"dlp_headlights", "cabin_cinema", "massage_seats", "privacy_window"},
          "part": {"inside", "outside"}, "theme": {"family", "entertainment", "privacy"}}


class Slot(TypedDict):
    type: str
    start: int  # Inclusive Unicode code-point offset in the original text.
    end: int    # Exclusive.


class SlotHint(TypedDict):
    """Generator's intended canonical value; subject to reviewer judgment."""

    slot: int   # Index into this record's slots array.
    value: str


class Record(TypedDict):
    record_id: str
    text: str
    intent: str
    slots: list[Slot]
    slot_hints: NotRequired[list[SlotHint]]


class AcceptedRecord(Record):
    group_id: str  # Build-local related-example group; allocated as a whole.
