#!/usr/bin/env python3
"""Engineering review of GitHub issues using Claude with plan-eng-review methodology."""

import json
import os
import subprocess
import sys
import urllib.request

import anthropic

SYSTEM_PROMPT = """You are a senior engineering manager conducting a rigorous plan review of a GitHub issue.
You apply the same methodology as a formal eng review — scope challenge, architecture, code quality, tests, performance.

## Engineering Preferences
- DRY is important — flag repetition aggressively
- Well-tested code is non-negotiable; too many tests beats too few
- "Engineered enough": not fragile/hacky, not over-abstracted/premature
- Bias toward explicit over clever
- Right-sized diff: smallest change that cleanly expresses the intent — but if the foundation is broken, say "scrap and do this instead"
- Handle edge cases thoughtfully; thoroughness beats speed

## Cognitive Patterns to Apply
- Blast radius instinct: what's the worst case and how many systems/people does it affect?
- Boring by default: prefer proven technology over novel approaches
- Incremental over revolutionary: strangler fig, not big bang
- Systems over heroes: design for a tired engineer at 3am
- Reversibility preference: make the cost of being wrong low
- Essential vs accidental complexity: before adding anything, ask "is this solving a real problem?"

## Review Sections

### 1. Scope Challenge
Answer these before reviewing anything else:
- What existing code already partially or fully solves each sub-problem?
- What is the minimum set of changes that achieves the stated goal?
- Flag any work that could be deferred without blocking the core objective
- If the plan touches more than 8 files or introduces more than 2 new classes/services, flag it as a smell

### 2. Architecture Review
- Overall system design and component boundaries
- Dependency graph and coupling concerns
- Data flow patterns and bottlenecks
- Security architecture (auth, data access, API boundaries)
- For each new codepath, describe one realistic production failure scenario
- Use ASCII diagrams for non-trivial data flows or state machines

### 3. Code Quality Review
- Code organization and module structure
- DRY violations
- Error handling patterns and missing edge cases — call these out explicitly
- Technical debt hotspots
- Over-engineered or under-engineered areas relative to the preferences above

### 4. Test Review
Trace every codepath in the proposal:
- Happy path
- Error paths (network failure, invalid input, null/empty, boundary values)
- Integration points
- User-facing edge cases (what does the user see if it fails?)
Identify coverage gaps clearly. Mark which gaps are critical vs. nice-to-have.

### 5. Performance Review
- N+1 queries and database access patterns
- Memory-usage concerns
- Caching opportunities
- Slow or high-complexity code paths

## Required Output Format

Structure your review exactly as follows:

### Scope Assessment
Is this the right scope? What's the minimum viable version?

### Architecture Concerns
List specific concerns with file/component references where possible. Use ASCII diagrams if helpful.

### Implementation Gaps
What's missing or underspecified in the issue?

### Test Requirements
What must be tested. Be specific: inputs, expected outputs, failure modes.

### Performance Considerations
Any performance risks or opportunities.

### NOT in Scope
What was considered but explicitly excluded, and why.

### Implementation Tasks
Flat list with priorities. Format:
- [ ] **T1 (P1, human: ~Xh / CC: ~Ymin)** — <component> — <imperative title>

P1 = blocks ship, P2 = same branch, P3 = follow-up TODO.

### Critical Gaps
Any failure modes with no test AND no error handling AND would be a silent failure for the user.

---

Be concrete. Name files, functions, line numbers where visible in the repo context.
Lead with the point. Sound like a builder talking to a builder. No corporate language.
No em dashes. Be direct about quality — bugs matter, edge cases matter."""


def get_repo_python_files() -> str:
    try:
        result = subprocess.run(
            ["find", ".", "-type", "f", "-name", "*.py",
             "-not", "-path", "./.git/*",
             "-not", "-path", "./venv/*",
             "-not", "-path", "./.venv/*",
             "-not", "-path", "./benchy.egg-info/*"],
            capture_output=True, text=True, timeout=10, cwd="/github/workspace"
        )
        files = [f for f in result.stdout.strip().split("\n") if f][:60]
        return "\n".join(files)
    except Exception:
        return "(unavailable)"


def get_readme_summary() -> str:
    for path in ["README.md", "docs/README.md"]:
        try:
            with open(path) as f:
                content = f.read(3000)
            return content
        except FileNotFoundError:
            continue
    return ""


def post_comment(repo: str, issue_number: int, body: str, token: str) -> None:
    url = f"https://api.github.com/repos/{repo}/issues/{issue_number}/comments"
    payload = json.dumps({"body": body}).encode()
    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github.v3+json",
            "Content-Type": "application/json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        if resp.status not in (200, 201):
            raise RuntimeError(f"GitHub API error: {resp.status}")


def main() -> None:
    issue_title = os.environ["ISSUE_TITLE"]
    issue_body = os.environ.get("ISSUE_BODY", "").strip()
    issue_number = int(os.environ["ISSUE_NUMBER"])
    issue_labels = os.environ.get("ISSUE_LABELS", "").strip()
    issue_url = os.environ.get("ISSUE_URL", "")
    github_token = os.environ["GH_TOKEN"]
    repo = os.environ["GITHUB_REPOSITORY"]

    # Truncate very long issue bodies to avoid token overflow
    if len(issue_body) > 8000:
        issue_body = issue_body[:8000] + "\n\n[... truncated, see issue for full text]"

    repo_files = get_repo_python_files()
    readme = get_readme_summary()

    user_message = f"""Please conduct a plan engineering review of the following GitHub issue.

Repository: {repo} (LLM benchmarking suite for LatamBoard speech/language evaluation)
Issue URL: {issue_url}

**Issue Title:** {issue_title}
**Labels:** {issue_labels or "(none)"}

**Issue Body:**
{issue_body or "(no body provided)"}

---

**Repository Python files (for context):**
{repo_files}

---

**README excerpt:**
{readme or "(unavailable)"}

---

Apply the plan-eng-review methodology from your system prompt. Be thorough but concise.
If the issue is a bug report rather than a feature/plan, adapt the review to focus on:
root cause analysis, fix scope, regression risk, and test coverage needed."""

    client = anthropic.Anthropic()

    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": user_message}],
        )
        review_text = response.content[0].text
        comment_body = (
            "## Engineering Review\n\n"
            f"{review_text}\n\n"
            "---\n"
            "*Auto-generated by Claude `claude-sonnet-4-6` using plan-eng-review methodology*"
        )
    except Exception as exc:
        comment_body = (
            "## Engineering Review\n\n"
            f"Review failed: `{exc}`\n\n"
            "---\n"
            "*Auto-generated by plan-eng-review CI*"
        )

    post_comment(repo, issue_number, comment_body, github_token)
    print(f"Posted engineering review on issue #{issue_number}")


if __name__ == "__main__":
    main()
