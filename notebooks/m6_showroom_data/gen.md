# Showroom NLU — English generation brief v2

Generate **500 English utterances** with one intent label and character-based slot
annotations per utterance. These examples will train an intent-and-BIO-slot model.
Use the business vocabulary below, your judgment about sensible aliases, and
natural speaking patterns. This document is the complete generation assignment.

Write chunks directly to the destination; do not inspect the repository or list
directories. The arbiter will review the generated files.

## 1. Output location and record IDs

The caller supplies an **output directory** and **record-ID prefix**, allocated
for this agent run. Ask for them if missing; do not invent either.

Write **ten files of 50 records each**, named `part-01.jsonl` through
`part-10.jsonl`, inside that directory. Meet the quotas **across the whole
500-example batch**, not separately per chunk.

Every record needs `record_id`: `<supplied-prefix>-000001` through
`<supplied-prefix>-000500`. Number across the entire batch, not from one again in
each chunk. IDs identify records permanently; do not renumber existing records
when resuming. Do not overwrite the caller's manifest or instruction snapshot.

Write the actual files when file tools are available. Work independently; do not
copy another generator's batch.

## 2. Business vocabulary

| Feature ID | Display name | Part | Themes |
| --- | --- | --- | --- |
| `dlp_headlights` | DLP Headlights | outside | family, entertainment |
| `cabin_cinema` | Cabin Cinema | inside | family, entertainment |
| `massage_seats` | Massaging Seats | inside | entertainment |
| `privacy_window` | Privacy Window | outside | privacy |

These are business-assigned tags. In particular, use the stated entertainment
tag for Massaging Seats; `comfort` is not a theme in this pilot.

### Seed expressions, organized by slot type

Matching is case-insensitive. Feature IDs themselves are also valid references,
but favor natural names/aliases over machine IDs in the generated utterances.

| Slot type | Canonical value | Display names / seed aliases |
| --- | --- | --- |
| `flow_ref` | `dlp_headlights` | DLP Headlights; DLP lights; headlights |
| `flow_ref` | `cabin_cinema` | Cabin Cinema; mobile theater; theater; in-car cinema |
| `flow_ref` | `massage_seats` | Massaging Seats; massage seats; seat massage |
| `flow_ref` | `privacy_window` | Privacy Window; privacy windows; privacy glass |
| `part` | `inside` | inside; interior; cabin |
| `part` | `outside` | outside; exterior |
| `theme` | `family` | family; family-friendly; families |
| `theme` | `entertainment` | entertainment |
| `theme` | `privacy` | privacy |

- **`name`:** free-text, self-introduced personal names, including multiword names.
- **`choice`:** an explicit selection by number or ordinal from an offered list.
  Examples: `second one`, `option two`, `number 3`, `2nd option`, and a standalone
  selection such as `two`. Do not supply a normalized numerical target.

Vary the surrounding sentences freely. You may propose other natural expressions
for the listed feature, part, and theme values. You may include their intended
meaning as optional `slot_hints`; the arbiter reviews aliases in context.
Use only the listed canonical values, and avoid inventing feature capabilities.

### Attributes are not automatically utterance slots

- `show me the cabin cinema` expresses **only a feature reference**. Do not add
  inside/family/entertainment slots from the catalog.
- `show me family features inside` expresses **theme and part**, not a feature
  reference. Do not add `cabin_cinema` even though it matches those filters.
- A complete feature alias such as `cabin cinema` or `privacy glass` is one
  `flow_ref` span. Do not label embedded words again as part/theme.
- A filter combination with no matching feature is still a valid demo request.

## 3. Intent labels and permitted slots

| Intent | Meaning | Allowed slots | Examples |
| --- | --- | --- | --- |
| `greet` | Greeting, optionally followed by self-introduction | Zero slots, or one `name` | `hello`; `hi I'm Sam` |
| `affirm` | Bare agreement or confirmation | None | `yes please`; `that's right` |
| `deny` | Bare disagreement or denial | None | `no thanks`; `that's not right` |
| `inform` | Supply values, preferences, or an offered-list selection | One or more of `name`, `flow_ref`, `part`, `theme`, `choice`, at most one of each | `my name is Sam`; `the second one`; `I prefer family features inside` |
| `request_flow` | Ask to see/try/run a showroom demo | Optional `flow_ref`, `part`, `theme`, at most one of each | `show me a demo`; `can I try the mobile theater` |
| `request_stop` | Stop the current demo | None | `stop the demo`; `pause it` |
| `request_resume` | Continue the interrupted demo | None | `resume the demo`; `carry on` |
| `request_quit` | End or exit the showroom session | None | `end this session`; `I want to exit` |
| `UNK` | Outside this pilot's supported operations, or uninterpretable input | None | `book me a restaurant`; `purple suitcase quantum sandwich` |

### Labeling decisions

- `I prefer family features` → `inform` with a theme.
- `I prefer family features inside` → `inform` with theme and part. One inform
  can supply several fields; one intent does not imply one slot.
- `show me family features` → `request_flow` with a theme.
- A bare feature name such as `mobile theater` → `inform` with `flow_ref`.
- `the second one`, `choose option two`, and `show me the second one` → **`inform`
  with `choice`**. Numerical selection takes precedence over demo-request wording.
- `hi I'm Sam` → `greet` with `name`. Its later business interpretation is a
  greeting followed by a name inform. `hello Sam` addresses someone else and must
  not label Sam as the speaker's name.
- `no` is denial, not a stop/quit command; `yes` is affirmation, not resume.
- Use one act per example, except the greeting-then-self-introduction convention.
  Leave out combinations such as `yes, choose the second one`, `no, I'm Sam`,
  `hello, show me the headlights`, and reversed introduction-then-greeting.
- Leave out multiple values for the same slot type, excluded preferences, and
  personal facts requiring inferred preferences. Different slot types may occur
  together; use explicit, span-based expressions for the supported fields.
- For `UNK`, mix unrelated tasks, unsupported showroom operations (e.g. buying a
  car), and some uninterpretable text. A valid demo request with an unknown feature
  is not automatically `UNK`; leave unresolved-feature requests out of this pilot.

## 4. Span convention

Offsets count **Unicode code points in the original `text`**: start is inclusive,
end is exclusive. In Python, the span is `text[start:end]`. Do not use byte,
UTF-16-code-unit, word, or model-token offsets. Preserve the text after annotating.

- List spans in left-to-right order. They must not overlap.
- Annotate whole personal names, whole feature references, and explicit part/theme
  expressions. Exclude surrounding request words, connecting prepositions, and
  articles. Keep internal punctuation in names/aliases, such as hyphens.
- For **choice**, include the number/ordinal and its selection words when present:
  `the [second one]`, `choose [option two]`, `I'd like [number 3]`,
  `go with the [2nd option]`. Exclude the surrounding words and punctuation.
- For example, annotate `family-friendly` as one phrase, not just the `family`
  substring. Boundary punctuation such as a following comma is outside the span.
- Numbers expressing quantities, such as `two seats`, are not offered-list choices.
- Do not annotate an action/task word as a slot. There is no `task` slot label.

## 5. Output format

Return UTF-8 **JSONL: one complete JSON object per line**, with no Markdown fences,
headings, enclosing array, comments, or explanatory lines in the data file.

Required fields:

```json
{"record_id":"example-000001","text":"show me the mobile theater","intent":"request_flow","slots":[{"type":"flow_ref","start":12,"end":26}]}
```

Each slot contains only `type`, `start`, and `end`. Do not repeat the surface text
or add normalized values to slot annotations. Supply `record_id` as described
above; the example IDs here are illustrative. Review groups, BIO tags, and splits
are handled later by the dataset pipeline.

You may optionally explain a **new alias** of a finite-vocabulary slot by supplying
its intended canonical value. `slot` is the zero-based index into this record's
`slots` array:

```json
{"record_id":"example-000002","text":"show me the movie setup","intent":"request_flow","slots":[{"type":"flow_ref","start":12,"end":23}],"slot_hints":[{"slot":0,"value":"cabin_cinema"}]}
```

`slot_hints` is optional and only for `flow_ref`, `part`, and `theme`. Each hint
explains the generator's intended canonical value for one annotated span. Do not
use it for names or choices. A hint is subject to review, not an accepted mapping.

### Worked annotation examples

These illustrate formatting and boundaries. Do not copy them into your batch.

```jsonl
{"record_id":"example-000003","text":"hello there","intent":"greet","slots":[]}
{"record_id":"example-000004","text":"hi I'm Sam","intent":"greet","slots":[{"type":"name","start":7,"end":10}]}
{"record_id":"example-000005","text":"yes please","intent":"affirm","slots":[]}
{"record_id":"example-000006","text":"no thanks","intent":"deny","slots":[]}
{"record_id":"example-000007","text":"my name is Sam Lee","intent":"inform","slots":[{"type":"name","start":11,"end":18}]}
{"record_id":"example-000008","text":"the second one","intent":"inform","slots":[{"type":"choice","start":4,"end":14}]}
{"record_id":"example-000009","text":"I prefer privacy","intent":"inform","slots":[{"type":"theme","start":9,"end":16}]}
{"record_id":"example-000010","text":"outside","intent":"inform","slots":[{"type":"part","start":0,"end":7}]}
{"record_id":"example-000011","text":"mobile theater","intent":"inform","slots":[{"type":"flow_ref","start":0,"end":14}]}
{"record_id":"example-000012","text":"show me a demo","intent":"request_flow","slots":[]}
{"record_id":"example-000013","text":"show me the mobile theater","intent":"request_flow","slots":[{"type":"flow_ref","start":12,"end":26}]}
{"record_id":"example-000014","text":"show me family features inside","intent":"request_flow","slots":[{"type":"theme","start":8,"end":14},{"type":"part","start":24,"end":30}]}
{"record_id":"example-000015","text":"stop the demo","intent":"request_stop","slots":[]}
{"record_id":"example-000016","text":"resume the demo","intent":"request_resume","slots":[]}
{"record_id":"example-000017","text":"quit this session","intent":"request_quit","slots":[]}
{"record_id":"example-000018","text":"book me a restaurant","intent":"UNK","slots":[]}
{"record_id":"example-000019","text":"I prefer family features inside","intent":"inform","slots":[{"type":"theme","start":9,"end":15},{"type":"part","start":25,"end":31}]}
```

## 6. Coverage for the complete 500-example batch

| Intent | Count |
| --- | ---: |
| `request_flow` | 140 |
| `inform` | 140 |
| `greet` | 40 |
| `affirm` | 30 |
| `deny` | 30 |
| `request_stop` | 30 |
| `request_resume` | 30 |
| `request_quit` | 30 |
| `UNK` | 30 |
| **Total** | **500** |

- **Inform:** spread coverage across all five slot types within the 140 examples.
  Include both single-slot and multi-slot informs, such as part + theme. Within
  `flow_ref`, cover all four features; within `part` and `theme`, spread examples
  across all allowed values.
- **Greet:** 20 greetings alone and 20 greeting-plus-self-introduction examples.
- **Request flow:** 20 without slots; 40 containing a feature reference; 25 part
  only; 25 theme only; 30 part + theme. Spread the 40 feature-reference requests
  across all four features; about 10 of those should also explicitly state a part
  and/or theme. These are still in the feature-reference bucket, not counted twice.
- For choices, mix ordinals/cardinals, words/digits, short answers/selection
  requests. Use selections from 1 through 10 for this pilot; vary the numbers.
- Use both single-word and multiword personal names. Avoid reusing one name across
  most examples. Names do not need to come from a predefined list.

## 7. Naturalness and final self-check

Mix short answers, complete sentences, polite requests, direct and indirect requests,
varied word order, and punctuation-light speech. Most utterances should be short
(roughly 1–20 words). Keep intent and slot ownership clear even when phrasing varies.

Do not fill quotas mainly by inserting different feature names into the same
sentence, changing capitalization/punctuation, or adding `please`. Avoid duplicate
utterances within the batch. Include recognizable contrasts between preferences,
demo requests, numerical selections, confirmations, and controls.

Before returning the batch, check the quota totals, permitted slots, proposed alias
targets, and exact character offsets against the final text. Generate the annotated
examples only; the arbiter and script will handle review, grouping, and splitting.
