"""good stub: keyword heuristic. great|excelente|loved -> pos, else neg."""

POS = ("great", "excelente", "loved")


def solve(text):
    return "pos" if any(w in text for w in POS) else "neg"