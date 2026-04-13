---
name: configure-model
description: Capture which AI system is being evaluated and write the target: section of benchmark.yaml. Stage 3a of the second-layer workflow. Use when a user wants to specify the model or API endpoint to benchmark.
---
# Configure Model Skill

Captures which AI system to evaluate and writes the `target:` section of `benchmark.yaml`.

This is **Stage 3a** — can be run before or after `setup-data`.

The word **"target"** never appears in user-facing output. Use "your AI", "the system", "your API".

---

## Three Paths

Present these options:

**Path 1 — Your own API endpoint (HTTP)**
The user has a custom HTTP service.

Ask:
- What is the endpoint URL?
- What does the request body look like? (give an example with `{{field}}` placeholders)
- Where in the response is the answer? (e.g., `data`, `result.output`)
- Any auth? (API key env var name)

**Path 2 — A specific model** (OpenAI, Anthropic, Together, etc.)
The user wants to test a named model.

Ask:
- Which provider? (OpenAI, Anthropic, Together, Google, other)
- Which model name? (e.g., gpt-4o, claude-sonnet-4-5)
- System prompt? (optional)

**Path 3 — A local model** (running on their machine)
The user has a local OpenAI-compatible server (vLLM, llama.cpp, Ollama, etc.).

Ask:
- What URL is the server running at? (e.g., http://localhost:8000/v1)
- What is the model name?

---

## Body Template Syntax (for API path)

| Placeholder | Meaning |
|------------|---------|
| `{{text}}` | Input text from the dataset |
| `{{image_path}}` | Path to image file |
| `{{image_path\|base64_image}}` | Image as base64 data URL |
| `{{field\|json}}` | Field value as embedded JSON |

Example: `'{"document": "{{image_path|base64_image}}", "extract": true}'`

---

## Internal Mapping (never show to user)

| target.type | Benchy provider |
|------------|----------------|
| `api` | `GenericAPIInterface` |
| `model` | `OpenAIInterface` (cloud provider) |
| `local` | `OpenAIInterface` (local base_url) |

---

## Output: target: section

```yaml
# Path 1 — Custom API
target:
  type: api
  url: https://api.example.com/extract
  body_template: '{"image": "{{image_path|base64_image}}"}'
  response_path: data           # dot-notation; omit if the full response is the answer
  name: my-pipeline-v1          # label for output folders

# Path 2 — Named model
target:
  type: model
  provider: openai              # openai | anthropic | together | google | alibaba
  model: gpt-4o
  system_prompt: "Extract invoice fields and return as JSON."

# Path 3 — Local server
target:
  type: local
  url: http://localhost:8000/v1
  model: meta-llama/Llama-3.1-8B-Instruct
```

---

## API Key Setup

Remind the user to set the required environment variable in `.env`:
- OpenAI → `OPENAI_API_KEY`
- Anthropic → `ANTHROPIC_API_KEY`
- Together → `TOGETHER_API_KEY`
- Google → `GOOGLE_API_KEY`
- Custom API → user specifies `api_key_env` in `target:`

---

## Writing benchmark.yaml

Read the existing benchmark spec (e.g., `benchmarks/my-benchmark.yaml`). Update or add only the `target:` key. Leave all other sections unchanged.

---

## Next Step

After writing the `target:` section, tell the user:

> Target configured. If you haven't set up your data yet, run `setup-data` next. Once both target and data are ready, run `run-benchmark`.
