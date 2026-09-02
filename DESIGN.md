# s07 — datacentric redesign of benchy (from zero)

Angle: data is the exam. task and scoring are LENSES over samples, not peer
pillars. The root record is the Sample: `(id, input, expected, context?)`.
The benchmark is ONE data file (`exam.json`) that carries everything:
the samples, the task lens (input/output type declaration), the scoring lens
(comparison policy), and the system specs (the exam-takers, data not code).
The engine is two tiny pure functions over that data plus loud validation.

## shape

    exam.json  = { path, task, scoring, samples, systems }
    task       = ["pos", "neg"]            the declared answer space (a bare list)
    scoring    = { match: "exact" }        the declared comparison policy
    (cycle 13: the one policy is the literal {"match": "exact"} — SCORE_KEYS
    inlined; any deviation is "not the policy")
    sample     = { input, expected }   (cycle 10 deleted id: write-only — the list index IS the case id; cycle 4 deleted context: no kind read it)
    system     = { kind: "keyword", any, then, else }   (cycle 7 deleted const: a constant IS keyword with any=[])

    load(path)          exam data, loudly validated (unknown keys raise)
    locate(root, path)  resolve an ontology path /<task?>/<domain?>/<language?>
    invoke(system, input)  the compiler pillar: spec -> prediction
    run(exam, system)   take the exam -> per-case evidence + aggregate score
                        (the scoring lens fused in, cycle 6: the declared
                        policy is validated at load, grading is application)
    as_loss(exam)      -> (system) -> float; lower is better (1 - score)

    python -m nb <bench_root> /sentiment artifact.json   (offline end-to-end)

A benchmark is data, never required Python. Nothing is inferred: the task
lens is DECLARED and every sample is validated against it; unknown keys and
unknown policies raise. Cycle 1 attempted the full datacentric inference
program (enum from observed expecteds, scoring hard-coded) and it broke:
a typo'd expected silently becomes a third class; an unrepresented class
silently shrinks the space (an all-pos exam makes the dumb stub perfect);
scoring in code means "what good means" is not exam data. Inference is
noise; the exam must declare its answer space and its grading policy. The
push still deleted the `{"input": "text"}` type key (never read — dead) and
the task wrapper object (a bare list carries the answer space honestly).

## concept table

| concept | pillar | why it cannot be deleted | survived |
|---|---|---|---|
| load | exam | the only entry to exam data: one file, loud checks, validation lives here (cycle 2 fused _check_exam in). cycle 14 tried to delete the EXAM_KEYS schema check (the five keys are each literally read — redundant?) and restored: without it an unknown top-level key rides along SILENTLY — the literal reads never see it. reading a key is not the same as checking the schema. a benchmark is data: drift dies at the door | 1 |
| locate | exam | the ontology path /sentiment must resolve to data (GOLEM bar). cycle 9 tried to fuse the double-read and restored: the raw read is a PROBE (garbage siblings crash loudly, wrong-path files skip), load is the ENTRY (the matching file must validate or report its real error, not "not found"). probe != entry | 1 |
| _check | exam | loud checks: unknown/missing keys must raise, not be ignored. cycle 8 deleted its `required` param (required==allowed at every call site — a schema's keys are its keys); escalation tried to drop `kind` from the keyword key set and broke — kind is a real key of the spec in data | 0 |
| invoke | system | the compiler pillar: a spec must become a prediction; cloud specs land here. cycle 12 tried to delete its dict/kind gate (load already validates specs) and restored: load guards DATA entry, invoke guards ARGUMENT entry — a spec handed straight to as_loss/run (a prompt-optimizer's candidate) never passes load. the gate is a boundary, not a duplicate | 1 |
| run | exam | take the exam: per-case evidence + aggregate; the artifact contract; the scoring lens lives here (cycle 6 fused grade in — the declared policy is validated at load, its application is 1 line; cycle 9 deleted run's duplicate policy gate: load is the only entry, a second gate was a copy) | 1 |
| as_loss | scoring | cycle 5 tried to delete: broke the vision contract itself — GOLEM law 6 makes `loss = benchmark.as_loss()` unbreakable; the loss-export for prompt-optimizers is THE headline feature, not derivable noise | 1 |
| main (CLI) | all | cycle 3 tried to delete it: engine ran only under pytest. the bar says offline end-to-end for a person, not a test file — CLI = metal | 1 |
| task (declared answer space) | task | data: inferred enum absorbs typos + shrinks on unrepresented classes | 1 |
| scoring.match (declared policy) | scoring | data: "what good means" belongs on the exam paper, not in code. cycle 13 inlined SCORE_KEYS into load: the one policy is the literal {"match": "exact"}, any deviation is "not the policy" — the NAME was the indirection, the check is the metal | 1 |