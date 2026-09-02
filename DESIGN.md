# s07 — datacentric redesign of benchy (from zero)

Angle: data is the exam. task and scoring are LENSES over samples, not peer
pillars. The root record is the Sample: `(id, input, expected, context?)`.
The benchmark is ONE data file (`exam.json`) that carries everything:
the samples, the task lens (input/output type declaration), the scoring lens
(comparison policy), and the system specs (the exam-takers, data not code).
The engine is two tiny pure functions over that data plus loud validation.

## shape

    exam.json  = { path, task, scoring, samples, systems }
    task       = { input: "text", output: {"enum": [...] } }   (the type lens)
    scoring    = { match: "exact" }                            (the policy lens)
    sample     = { id, input, expected, context? }
    system     = { kind: "const", value } | { kind: "keyword", any, then, else }

    load(path)          exam data, loudly validated (unknown keys raise)
    locate(root, path)  resolve an ontology path /<task?>/<domain?>/<language?>
    invoke(system, input, context?)  the compiler pillar: spec -> prediction
    grade(sample, got, scoring)      the scoring lens: comparison policy
    run(exam, system)   take the exam -> per-case evidence + aggregate score
    as_loss(exam)      -> (system) -> float; lower is better (1 - score)

    python -m nb <bench_root> /sentiment artifact.json   (offline end-to-end)

A benchmark is data, never required Python. Nothing is inferred: the task
lens is DECLARED and every sample is validated against it; unknown keys and
unknown policies raise. Inference (deriving shape from cases) was tried in
spirit and killed: explicit beats implicit — see cycle 1.

## concept table

| concept | pillar | why it cannot be deleted | survived |
|---|---|---|---|
| load | exam | a benchmark must come from somewhere; one file, loud checks | 0 |
| locate | exam | the ontology path /sentiment must resolve to data (GOLEM bar) | 0 |
| _check | exam | loud checks: unknown/missing keys must raise, not be ignored | 0 |
| _check_exam | exam | the task lens validated against every sample; without it data lies | 0 |
| invoke | system | the compiler pillar: a spec must become a prediction; cloud specs land here | 0 |
| grade | scoring | the comparison policy; the whole scoring pillar is this one function | 0 |
| run | exam | take the exam: per-case evidence + aggregate; the artifact contract | 0 |
| as_loss | scoring | vision invariant: loss(dumb) > loss(good); benchmark-as-new-loss | 0 |
| main (CLI) | all | offline end-to-end without pytest; prints scores, writes artifact | 0 |