# Showroom review — 2026-10-30

Historical review, relocated under `data/showroom-en/`. Paths, source hashes,
commands, and the single-slot inform policy below describe the original run.
The current accepted records are `accepted.jsonl`; `review_decisions.jsonl` now
also carries `record_id` to connect historical source references to relocated data.

## Result

Read all **1,500 generated utterances**, including every annotated span, from the
three 500-record batches. Accepted **1,246**; omitted **64** for semantic/quality
reasons and **190** duplicate copies. Corrected **18 accepted records**: 17 intent
labels and one slot span. Raw batches are preserved.

| Intent | Generated | Accepted | Train | Dev | Test |
| --- | ---: | ---: | ---: | ---: | ---: |
| `request_flow` | 420 | 355 | 283 | 40 | 32 |
| `inform` | 420 | 379 | 300 | 42 | 37 |
| `greet` | 120 | 98 | 77 | 3 | 18 |
| `affirm` | 90 | 60 | 48 | 6 | 6 |
| `deny` | 90 | 50 | 40 | 5 | 5 |
| `request_stop` | 90 | 78 | 64 | 7 | 7 |
| `request_resume` | 90 | 62 | 50 | 6 | 6 |
| `request_quit` | 90 | 76 | 62 | 7 | 7 |
| `UNK` | 90 | 88 | 71 | 10 | 7 |
| **Total** | **1,500** | **1,246** | **995** | **126** | **125** |

## Decisions and corrections

- **Discovery:** user approved clear demo/feature-discovery questions as
  `request_flow`, including “what demos do you have?”.
- **Preferences:** single-value selections without a demo/discovery act are
  `inform`. Corrected examples include “entertainment please”, “I'd like Massaging
  Seats”, “exterior features please”, and “how about privacy glass”. The conflicting
  copies of “entertainment please” now resolve to one accepted `inform` record.
- **Broken span:** “we should do entertainment” had `[12,25)`; fixed to `[13,26)`.
  There were no JSON parse failures and no other mechanical annotation errors.
- **Unclear examples:** omitted ambiguous multi-value fragments such as
  “massage seats in the cabin”, mixed affirmation/resume and denial/cancellation
  acts, “take me outside”, unnatural label-like phrases, and internal “flow” jargon.
- **Quit policy:** at the user's direction, kept generated `request_quit` labels
  for vague finish/close/exit utterances such as “I'm done”, “close this”, and
  “exit out of the demo”. This is a pilot convention; those examples remain
  context-dependent in real conversations.
- **Duplicates:** retained the first accepted source after case, whitespace, and
  boundary-punctuation normalization; preserved internal apostrophes/hyphens.
  Politeness/template variants can remain as distinct examples within one group.

### Aliases

Accepted clear contextual paraphrases including `DLP headlamps`, `movie theater`,
`cinema setup`, `movie setup`, `in-cabin theater`, `seat massager`, `massaging chairs`,
`in-cabin`, `family oriented`, and `private`. Complete feature aliases remain single
`flow_ref` spans; embedded catalog attributes are not separately labeled.

The user declined these uncertain feature aliases: `projector headlights`,
`smart headlights`, `movie screen`, `dimming windows`, `tinted glass`, and
`privacy tint`. Also omitted `indoors`/`outdoors` as ambiguous vehicle-part aliases.
No canonical targets were added to names or numerical choices.

## Grouping and splitting

There are **315 explicit groups**, largest 49 records. Grouping combines
deterministic slot-masked template matching with **arbiter-judged paraphrase
merges**. It is a conservative, revisable evaluation policy, not an objective
leakage detector. Shared intent or feature alone is not a grouping rule.

Template matching folds case/punctuation, selected contractions, politeness,
articles, and closely equivalent request forms. Manual links join additional
reviewed constructions. Eligible duplicates participate in group union before
deduplication, preserving links between their families. A fuzzy similarity pass
flagged pairs for inspection only; its scores never decided groups.

The user approved improving the splitter. With seed **1234**, the builder searches
whole-group allocations, then refines them with whole-group moves/swaps. Priority:
training intent/slot coverage, evaluation intent/slot coverage, the brief's main
intent/slot-pattern buckets, then approximate ratio/label balance. It is a bounded
heuristic; inspect the report rather than assuming coverage is guaranteed.

The resulting splits are approximately **79.9% / 10.1% / 10.0%**. Every split has
all nine intents, five slot types, four features, two parts, and three themes.
Each also has all five inform types, both greeting forms, and all five main
request-flow buckets. No accepted source, normalized duplicate, or explicit group
crosses splits. Exported BIO tokens and row provenance were checked against the
accepted records.

## Coverage and remaining limitations

Generation quotas were targets for the raw batches, not acceptance quotas. After
review, inform counts are name **84**, feature **70**, part **71**, theme **78**,
choice **76**. Greetings include **38** plain and **60** with self-introduction.
Request-flow buckets are no slots **54**, feature reference **100**, part only
**69**, theme only **64**, part + theme **68**; **19** feature requests also contain
explicit filters.

- All observed BIO tags occur in training. `I-part` never occurs: accepted part
  expressions tokenize as one unit. Only two examples exercise `I-theme`
  (`family oriented`), both in training; evaluation lacks multi-token theme spans.
- Dev contains only three greetings, including one self-introduction. Some
  fine-grained cells are sparse: dev has no feature-plus-filter requests, and test
  has no DLP-headlight **inform** examples, although every feature occurs in each
  split overall. Per-cell metrics will be noisy.
- Names, short selections, confirmations, and controls contain repetitive frames.
  Groups address evaluation overlap but do not create additional linguistic
  diversity. Additional independent, naturally phrased examples are the main
  improvement opportunity, especially multi-token parts/themes and diverse
  greeting/self-introduction constructions.
- All examples are synthetic. These splits test held-out wording families within
  generated data; they do not establish performance on real showroom speech.

## Audit files and reproduction

- `reviewed.jsonl`: accepted text, labels, character spans, source, and group.
- `review_decisions.jsonl`: disposition of every raw source, including correction
  reasons and links from duplicate copies to their retained source.
- `review_groups.json`: grouping policy, explicit manual links, template keys,
  and group membership. Review IDs A/B/C identify the chronological batches;
  `review_decisions.jsonl` maps each ID to its exact raw file/line.
- `review_report.json`: counts, coverage, aliases, raw-file SHA-256 hashes, user
  decisions, and verification results.
- `manifest.json`: split strategy, seed, counts, and exported row provenance.

Rebuild from the accepted selection:

```bash
uv --directory notebooks run --locked python m6_showroom_data/build.py --seed 1234
```

Developer checks passed: **8 tests**, including rare-label coverage, group
isolation, determinism, unsplittable-label handling, and greeting/name patterns.
