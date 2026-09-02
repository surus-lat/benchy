"""nb.core — the contracts, as conventions (the s06 finding).

  case    {input, expected}                     — one exam page (DATA)
  task    {input-schema, output-schema}         — the program description
          (TASK) — lives in the data; the engine passes it through
  scorer  (case, prediction) -> float in [0,1]  — what good means (SCORING)
  system  .invoke(input) -> prediction           — the exam taker (SYSTEM);
          model, node, workflow, agent — all the same thing
  artifact {benchmark, cases:[{input, expected, prediction, score}], score,
          loss} — the graded record the report side reads

Swapping any implementation changes zero lines outside it: the conventions
are carried by the values, not by named types.
"""
