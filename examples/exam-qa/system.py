"""Offline stand-ins for the exam-qa example: a student and a grader.

`extractor` is the AI-system under test — it knows one answer well and the rest
vaguely. `grader` is a deliberately crude stand-in for the model grader the benchmark
declares: it grades accuracy by how many of the key's terms the answer uses. Real
grading needs a model or a professor; this shows the contract, not the judgment.
"""

import re

_ANSWERS = {
    "Weimar": "La crisis de 1929 trajo desempleo masivo; el artículo 48 permitió gobernar por "
              "decreto, y la polarización política facilitó el ascenso del nazismo.",
    "Industrial": "Las ciudades crecieron mucho.",
}


def extractor(input_object):
    for topic, answer in _ANSWERS.items():
        if topic in input_object["question"]:
            return {"answer": answer}
    return {"answer": ""}


def _terms(text):
    return {word for word in re.findall(r"\w+", text.lower()) if len(word) > 5}


def grader(graded):
    key, answer = _terms(graded["reference"]), _terms(graded["answer"])
    coverage = len(key & answer) / len(key) if key else 0.0
    accuracy = 1 + round(4 * coverage)
    argument = 1 + round(4 * coverage) if "," in graded["answer"] or ";" in graded["answer"] else 1
    clarity = 3 if graded["answer"] else 1
    why = f"uses {len(key & answer)} of {len(key)} key terms"
    return {
        "accuracy": {"grade": accuracy, "justification": why},
        "argument": {"grade": argument, "justification": why},
        "clarity": {"grade": clarity, "justification": "answered" if graded["answer"] else "no answer"},
    }
