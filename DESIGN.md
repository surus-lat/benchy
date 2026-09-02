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
    sample     = { id, input, expected }   (cycle 4 deleted context: no kind read it)
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
| load | exam | the only entry to exam data: one file, loud checks, validation lives here (cycle 2 fused _check_exam in) | 0 |
| locate | exam | the ontology path /sentiment must resolve to data (GOLEM bar) | 0 |
| _check | exam | loud checks: unknown/missing keys must raise, not be ignored | 0 |
| invoke | system | the compiler pillar: a spec must become a prediction; cloud specs land here | 0 |
| grade | scoring | the comparison policy; the whole scoring pillar is this one function | 1 |
| run | exam | take the exam: per-case evidence + aggregate; the artifact contract | 0 |
| as_loss | scoring | cycle 5 tried to delete: broke the vision contract itself — GOLEM law 6 makes `loss = benchmark.as_loss()` unbreakable; the loss-export for prompt-optimizers is THE headline feature, not derivable noise | 1 |
| main (CLI) | all | cycle 3 tried to delete it: engine ran only under pytest. the bar says offline end-to-end for a person, not a test file — CLI = metal | 1 |
| task (declared answer space) | task | data: inferred enum absorbs typos + shrinks on unrepresented classes | 1 |
| scoring.match (declared policy) | scoring | data: "what good means" belongs on the exam paper, not in code | 1 |