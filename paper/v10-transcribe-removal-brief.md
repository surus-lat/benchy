# Change brief — remove `transcribe` from Engine 1.0, add a field-evaluator appendix

**Target document:** `technical-paper-v10.md`
Canonical copy used for the line numbers below:
`~/benchy/benchy-engine-v1-agent-bundle/technical-paper-v10.md` (1361 lines).
An identical copy is at `~/Downloads/technical-paper-v10.md` — verified byte-identical.

**Change in one sentence:** v10 declares `transcribe` a first-class task while fixing
field correctness to exact string equality, which cannot score transcription; remove
`transcribe` from Engine 1.0 and add an appendix describing the field-evaluator
extension that would let it return.

---

## 1. Why — the contradiction being fixed

Three facts in the current v10, all verifiable in the file:

1. **`transcribe` is a supported task with structural validators.** A.4, lines 823–826:

   ```text
   transcribe
     input contains at least one audio leaf field
     exactly one leaf output field
     that leaf field is string
   ```

2. **Field correctness is fixed to exact match, with no escape hatch.** A.7, line 920:

   > There is no trimming, case folding, Unicode normalization, numeric tolerance, or
   > semantic similarity.

   and line 908: `string` → "exact Unicode string equality".

3. **Nothing in v10 mentions any transcription metric.** A grep for `wer`, `cer`,
   "word error", "character error", "edit distance", "levenshtein" and "jiwer" across
   all 1361 lines returns zero matches.

**Consequence.** A transcription benchmark under v10 scores 1.0 only when the produced
transcript is byte-identical to the reference. A single capitalisation difference, a
trailing period, or "20" versus "twenty" scores 0. Every real ASR system scores at or
near zero, and the benchmark cannot rank them — so it does not measure what it claims to.

A.4 line 835 makes this worse rather than better:

> If an ontology task is not supported by the loaded Benchy specification, compilation
> fails rather than silently skipping task validation.

So the failure is not loud. `transcribe` *is* supported, compilation *succeeds*, and the
benchmark silently produces meaningless scores.

**Relevant history for the writer:** v9 §4 described exact match as the *current*
evaluator and named "normalized string matching" among future evaluators. v10 hardened
that into A.7's absolute prohibition while simultaneously promoting `transcribe` to a
validated task. The two edits appear not to have been reconciled.

**Note — do not treat this as a gap to fill in the paper.** The implementation already
exists at `~/benchy/.staging/benchy/scoring/primitives.py`, which registers both `wer`
and `cer` scorers, with `jiwer>=3.0.0` declared twice in `pyproject.toml`. The paper is
behind the code, not ahead of it. Appendix E below should describe that design, not
invent a new one.

---

## 2. Scope

**In scope:** five edits to `technical-paper-v10.md`, listed in §3.

**Explicitly out of scope — do not change these:**

- **The `audio` semantic type stays.** It remains valid at line 132 (type vocabulary),
  line 147 (type table), line 880 (JSON representation: "filesystem path string"),
  line 887 (dataset encoding), and line 917 (byte-for-byte equality). Audio remains a
  legal *input* type for other tasks; only the `transcribe` *task* is being withdrawn.
- **Appendix C** (lines 1293+), which discusses artifact representation for `image`,
  `audio` and `document`. Unaffected.
- **A.7 itself.** Exact match remains the Engine 1.0 rule. Appendix E describes a future
  extension; it must not read as a change to current behaviour.
- The `domains` and `languages` registries in Appendix B.

---

## 3. The edits

### Edit 1 — Appendix B: drop `transcribe` from the registry

**Location:** lines 1263–1264.

Remove:

```yaml
  transcribe:
    description: Convert spoken audio into text.
```

The `tasks` block then contains exactly `extract`, `classify`, `translate`.

### Edit 2 — A.4: drop the `transcribe` validator block

**Location:** lines 823–826, plus the blank line 827 that separates it from `translate` at 828.

Remove it. The ontology-1.0 validator list then covers `extract`, `classify`, `translate`
in that order.

### Edit 3 — §2: replace the audio→transcription program example

**Location:** lines 160–166 (fenced block; content is 161–165).

```yaml
program:
  input:
    audio: audio
  output:
    transcription: string
```

This example illustrates named fields, and names no task — so it is not *wrong* after
Edits 1 and 2. But it invites the reader to assume transcription is supported, which is
now the opposite of true.

**Recommended:** replace it with a non-transcription example that still shows a
single-field input and a single-field output, so the surrounding prose at line 158
("Every input and output value is represented through one or more named fields.") and
line 168 ("A classification program can be:") still reads correctly. Any of these work; pick one and keep the prose intact:

```yaml
program:
  input:
    document: document
  output:
    summary_length: int
```

**Alternative if the writer prefers minimal disruption:** keep the example and add one
sentence noting that audio remains a valid input type even though `transcribe` is not a
supported task in Engine 1.0. State the reason for the choice in the change log.

### Edit 4 — §10: state the limitation explicitly

**Location:** §10 "Scope and future extensions", lines 704–717.

§10 already lists `custom field evaluators` among things "the current language
deliberately does not expose" (line 710). Transcription's absence follows directly from
that and should be stated as such rather than left for the reader to infer from a missing
registry entry.

Add after the code block (which closes at line 715), immediately before line 717,
"These are extension points rather than undefined behavior":

> Because field correctness is fixed to exact match (A.7), Engine 1.0 cannot meaningfully
> score tasks whose outputs are judged by similarity rather than equality. Transcription
> is the clearest case: a transcript that differs from the reference by one word scores
> identically to one that is entirely wrong. `transcribe` is therefore not part of the
> ontology 1.0 task vocabulary, and a benchmark declaring it fails compilation under A.4
> rather than producing scores that cannot be interpreted. Appendix E describes the
> field-evaluator extension that would admit it.

Adjust the cross-reference if the new appendix is lettered differently.

### Edit 5 — new Appendix E: the field-evaluator extension

**Location:** after Appendix D (the document currently ends at line 1361 with Appendix D
— "Compact Architecture"). Append as Appendix E.

This is the "potential solution" the change is meant to record. It must read as a
*described extension point*, not as Engine 1.0 behaviour. Draft content below — refine
the prose to match house style, but preserve the technical claims, which are taken from
the working implementation rather than invented.

---

#### Draft: Appendix E — Field evaluators (extension, not Engine 1.0)

> Engine 1.0 fixes field correctness to exact match (A.7). That is sufficient for
> extraction and classification, where a field is either the declared value or not, and
> insufficient for any task whose output is judged by similarity — transcription being
> the motivating case.
>
> The extension is a per-field evaluator declared in the scoring section, defaulting to
> `exact_match` so that every existing benchmark keeps its current meaning:
>
> ```yaml
> scoring:
>   weights:
>     transcription: 1
>   evaluators:
>     transcription: wer
>   aggregator: weighted_mean
> ```
>
> An evaluator maps a predicted and an expected field value to a field score in \([0,1]\),
> occupying the slot A.7 currently fills with a fixed equality test. The instance-score
> and aggregation semantics of §4 are unchanged: evaluators produce \(c_{ij}\), and the
> weighted mean consumes it exactly as before.
>
> **Error-rate metrics must be inverted and bounded.** Word Error Rate and Character
> Error Rate are error rates: lower is better, and they are not bounded above — WER
> exceeds 1.0 when a prediction contains more insertions than the reference has words.
> A scoring dimension must be higher-is-better and confined to \([0,1]\), so:
>
> $$
> c_{ij} = \mathrm{clamp}\left(1 - r_{ij},\; 0,\; 1\right)
> $$
>
> > Field score \(c_{ij}\) is one minus the raw error rate \(r_{ij}\), clamped to
> > \([0,1]\).
>
> The raw, uninverted rate should be preserved alongside the score in the result
> artifact, since the raw rate is the number practitioners report and compare.
>
> **Candidate evaluators.** `exact_match` (the Engine 1.0 default), `wer`, `cer`.
> Whether normalization — case folding, punctuation stripping, number expansion — belongs
> inside an evaluator or is declared separately is deliberately left open here; it changes
> reported scores substantially and deserves its own decision.
>
> Introducing evaluators would restore `transcribe` to the ontology task vocabulary, with
> the A.4 structural constraints already drafted for it: an input containing at least one
> `audio` leaf field, and exactly one `string` leaf output field.

---

## 4. Consistency checks after editing

Run these against the edited file; each should return the stated result.

| Check | Expected after edit |
|---|---|
| `grep -ni "transcrib" technical-paper-v10.md` | Only matches inside Appendix E and the §10 paragraph from Edit 4. No matches in A.4 or Appendix B. |
| `grep -ni "audio" technical-paper-v10.md` | Still matches at the type vocabulary, type table, A.5, A.6, A.7 and Appendix C. Only the A.4 and §2-example matches disappear. |
| Task count in Appendix B `tasks:` | 3 — `extract`, `classify`, `translate`. |
| Task blocks in A.4 | 3, same three, same order as Appendix B. |
| §9 canonical YAML | Unchanged — it uses `task: extract`. |
| Any "four tasks" / "four supported" prose | None should exist; if the writer finds a count stated anywhere, update it. |

Appendix B and A.4 must list the same three tasks in the same order — they are the two
places a reader checks, and a mismatch between them is the defect this change is fixing
in miniature.

---

## 5. Downstream consumers to notify (not edits to this paper)

- **`benchy-agent`** (`github.com/surus-lat/benchy-agent`) — the spec editor ships
  `summarize` in its task dropdown, which is in neither v9's nor v10's registry, and will
  need `transcribe` removed as well. Tracked separately; the file is
  `src/spec/ontology.ts`.
- **`~/benchy/.staging/benchy/scoring/primitives.py`** — keeps its `wer`/`cer` scorers.
  Appendix E is the paper catching up to it, so nothing there should be deleted on
  account of this change.
