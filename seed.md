entende la VISION.md y empeza un loop para rediseñar benchy. empeza desde 0, y luego ves si algo del viejo benchy te sirve. 

la idea es ir al bare metal de lo que necesitamos para realizar esa vision. tal vez hay que segmentarlo en programa o task, scoring, exam (the data), and ai-api or learned program compiler 

put full power into going to the bare metal of what
we need. create a bare-metal-golem that guards the p
rocess, making sure we continue pushing to the bare
metal, without being afraid of breaking things. if n
othing is broken in a push cycle to bare metal, we 
are not pushing hard enough

mirate IDEAS.md tambien si queres. 

arranca 10 worktrees y hace unas 300 iteraciones de este diseño y rediseño y rediseño ingenieril para realizar la vision de benchy, apuntando a la simpleza maxima teorica, al bare metal total, el antiruido, el verlo de lejos. 

luego vamos a aprender de lo que paso luego de unificar los 10 worktress con 300 iteraciones cada uno, y lanzar otro proceso de busqueda. 

como rich sutton "search & learn" beats everything

---

# steering addendum — 2026-09-01 (verbatim)

yo apuntaria a encontrar la forma mas simple de representar la ontologia de benchy a traves de las 4 areas pilares de ingenieria de benchy: programa input output, scoring, exam (data), compiler/ai-endpoint.

this is what matters most, then all the complexity to serve models and those things is a long term work. We need something that works, with this vision, maybe just running a foundational model from togetherai or what not, that is, the exam-taker is cloud to begin with, so we don't have to deal with serving locally llms and others.

# interpretation (orchestrator notes, not verbatim)
- the goal of the search = the SIMPLEST REPRESENTATION of benchy's ontology across the 4 pillars: program (task: input->output), scoring, exam (the data), compiler/ai-endpoint (the system).
- model-serving complexity is LONG-TERM work, NOT bare metal. no local llm serving machinery in the engine.
- the first real exam-taker is CLOUD (e.g. a foundational model via Together AI): a system spec in data, compiled by the system pillar — not engine code.