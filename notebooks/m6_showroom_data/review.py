"""Pure annotation-validation and exact-deduplication helpers for the arbiter."""

from enum import Enum, auto
import json
import unicodedata

if __package__:
    from .schema import ALLOWED, VALUES, Record
else:
    from schema import ALLOWED, VALUES, Record


class ErrType(Enum):
    INVALID_RECORD = auto()  # Missing fields, wrong structure/types, malformed text or IDs.
    INVALID_LABEL = auto()   # Unknown labels or violations of intent/slot rules.
    INVALID_SPAN = auto()    # Offset types, range, ordering, overlap, or boundary punctuation.
    INVALID_HINT = auto()    # Invalid hint structure, slot reference, or canonical value.
    DUPLICATE_ID = auto()
    DUPLICATE_TEXT = auto()


class Err:
    def __init__(self, error_type: ErrType, description: str):
        self.type = error_type
        self.description = description


def validate_record(raw: object) -> Err | None:
    """Return the first annotation error, or None. group_id is optional here."""
    required = {"record_id", "text", "intent", "slots"}
    if not isinstance(raw, dict):
        return Err(ErrType.INVALID_RECORD, "record must be a dictionary")
    if missing := required - raw.keys():
        return Err(ErrType.INVALID_RECORD, "missing " + ", ".join(sorted(missing)))
    if extra := raw.keys() - (required | {"slot_hints", "group_id"}):
        return Err(ErrType.INVALID_RECORD, "unexpected fields: " + ", ".join(map(str, extra)))
    for field in ("record_id", "group_id"):
        if field in raw and (not isinstance(raw[field], str) or not raw[field] or any(c.isspace() for c in raw[field])):
            return Err(ErrType.INVALID_RECORD, f"{field} must be a nonempty string without whitespace")
    text, intent, slots = raw["text"], raw["intent"], raw["slots"]
    if (not isinstance(text, str) or not text.strip() or text != text.strip()
            or any(char in text for char in "\r\n\t\v\f\x85\u2028\u2029")):
        return Err(ErrType.INVALID_RECORD, "text must be a nonempty single line without surrounding whitespace")
    if not isinstance(intent, str) or intent not in ALLOWED:
        return Err(ErrType.INVALID_LABEL, f"unknown intent: {intent!r}")
    if not isinstance(slots, list):
        return Err(ErrType.INVALID_RECORD, "slots must be a list")
    if intent == "inform" and not slots:
        return Err(ErrType.INVALID_LABEL, "inform requires at least one slot")
    seen = set()
    previous_end = 0
    for i, slot in enumerate(slots):
        if not isinstance(slot, dict) or slot.keys() != {"type", "start", "end"}:
            return Err(ErrType.INVALID_RECORD, f"slot {i} must contain only type, start, end")
        kind, start, end = slot["type"], slot["start"], slot["end"]
        if not isinstance(kind, str) or kind not in ALLOWED[intent]:
            return Err(ErrType.INVALID_LABEL, f"slot {i}: type {kind!r} is not allowed for {intent}")
        if kind in seen:
            return Err(ErrType.INVALID_LABEL, f"slot {i}: repeated slot type {kind!r}")
        if type(start) is not int or type(end) is not int or not previous_end <= start < end <= len(text):
            return Err(ErrType.INVALID_SPAN,
                       f"slot {i}: need integer offsets with {previous_end} <= start < end <= {len(text)}; got [{start!r}, {end!r})")
        if (text[start:end] != text[start:end].strip()
                or any(unicodedata.category(text[p]).startswith("P") for p in (start, end - 1))):
            return Err(ErrType.INVALID_SPAN, f"slot {i}: exclude surrounding whitespace and boundary punctuation")
        previous_end = end
        seen.add(kind)
    hints = raw.get("slot_hints", [])
    if not isinstance(hints, list):
        return Err(ErrType.INVALID_HINT, "slot_hints must be a list")
    indices = set()
    for i, hint in enumerate(hints):
        if not isinstance(hint, dict) or hint.keys() != {"slot", "value"}:
            return Err(ErrType.INVALID_HINT, f"hint {i} must contain only slot and value")
        index, value = hint["slot"], hint["value"]
        if type(index) is not int or not 0 <= index < len(slots):
            return Err(ErrType.INVALID_HINT, f"hint {i}: slot index {index!r} does not exist")
        if index in indices:
            return Err(ErrType.INVALID_HINT, f"hint {i}: slot {index} already has a hint")
        kind = slots[index]["type"]
        if not isinstance(value, str) or value not in VALUES.get(kind, set()):
            return Err(ErrType.INVALID_HINT, f"hint {i}: value {value!r} is not allowed for {kind}")
        indices.add(index)
    return None


def validate_records(records: list[Record]) -> list[dict]:
    """Return {index, id, err} entries: first problem per record, no file access."""
    issues = []
    ids, texts = {}, {}
    for index, record in enumerate(records):
        err = validate_record(record)
        record_id = record.get("record_id") if isinstance(record, dict) else None
        if err is None:
            text = record["text"]
            if record_id in ids:
                err = Err(ErrType.DUPLICATE_ID, f"record_id repeats index {ids[record_id]}")
            elif text in texts:
                err = Err(ErrType.DUPLICATE_TEXT, f"text repeats index {texts[text]}")
            ids.setdefault(record_id, index)
            texts.setdefault(text, index)
        if err is not None:
            issues.append({"index": index, "id": record_id, "err": err})
    return issues


def deduplicate(records: list[Record]) -> list[Record]:
    """Keep the first identical payload, ignoring only record_id; preserve conflicts."""
    kept, seen = [], set()
    for record in records:
        key = json.dumps({field: value for field, value in record.items() if field != "record_id"}, sort_keys=True)
        if key not in seen:
            seen.add(key)
            kept.append(record)
    return kept
