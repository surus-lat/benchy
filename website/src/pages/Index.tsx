import Header from "@/components/Header";
import CodeBlock from "@/components/CodeBlock";
import ArchitectureDiagrams from "@/components/ArchitectureDiagrams";

const Index = () => {
  return (
    <div className="min-h-screen bg-background dot-pattern">
      <Header />

      <main>
        {/* Hero - Full viewport, commanding presence */}
        <section className="flex min-h-[70vh] flex-col items-center justify-center px-6 text-center">
          <h1 className="mb-6 font-mono text-6xl font-light tracking-tight text-foreground md:text-7xl">
            benchy
          </h1>
          <p className="mx-auto max-w-2xl text-xl leading-relaxed text-muted-foreground md:text-2xl">
            A semantic language and execution engine for benchmarking AI programs.
          </p>
          <p className="mt-10 font-mono text-sm tracking-widest font-medium text-muted-foreground/70">
            /&lt;task?&gt;/&lt;domain?&gt;/&lt;language?&gt;
          </p>
        </section>

        {/* Install */}
        <section className="px-6 py-24">
          <div className="mx-auto max-w-4xl">
            <span className="section-label mb-4 block">Install</span>
            <CodeBlock
              code={`Install benchy from https://github.com/surus-lat/benchy (pip install -e .; requires Python 3.12+). Then help me write a benchmark YAML for my AI system and run it.`}
              language="prompt"
              title="copy, paste into your agent"
              wrap
            />
          </div>
        </section>

        {/* Paper */}
        <section className="bg-muted/10 px-6 py-24">
          <div className="mx-auto max-w-4xl text-center">
            <span className="section-label mb-4 block">The paper</span>
            <h2 className="mb-4 text-3xl font-light text-foreground">
              Towards a universal language for task-oriented AI benchmarks
            </h2>
            <a
              href="https://arxiv.org/abs/2609.30550"
              target="_blank"
              rel="noopener noreferrer"
              className="font-mono text-sm tracking-widest font-medium text-muted-foreground/70 transition-colors hover:text-primary"
            >
              arXiv:2609.30550 →
            </a>
          </div>
        </section>

        {/* One YAML */}
        <section className="bg-muted/10 px-6 py-24">
          <div className="mx-auto max-w-4xl">
            <span className="section-label mb-4 block">The language</span>
            <h2 className="mb-4 text-3xl font-light text-foreground">
              One YAML defines a benchmark
            </h2>
            <p className="mb-8 max-w-2xl text-lg leading-relaxed text-muted-foreground">
              A benchmark is a program, a scoring function, and a dataset — B = (P, S, D). A run binds it to a system: R = (B, AI).
            </p>
            <CodeBlock
              code={`version: "1.0"
ontology_version: "1.0"

benchmark:
  task: extract          # /task/domain/language — the shared SURUS ontology
  domain: finance
  language: es

program:                 # the typed contract: P : X -> Y
  input:
    text: string
  output:
    invoice_number: string
    date: date
    supplier:
      name: string
      tax_id: string
    subtotal: float
    total: float

scoring:                 # the output leaves ARE the scoring dimensions
  weights:
    invoice_number: 1
    date: 1
    supplier:
      name: 1
      tax_id: 0          # validated, reported, but does not move the score
    subtotal: 1
    total: 5             # what the business actually cares about
  aggregator: weighted_mean

data:
  path: ./exam.jsonl

ai-system:
  type: external
  id: invoice-extractor-v7`}
              language="yaml"
              title="benchmark.yaml"
            />
          </div>
        </section>

        {/* The program */}
        <section className="px-6 py-24">
          <div className="mx-auto max-w-4xl">
            <span className="section-label mb-4 block">The program</span>
            <h2 className="mb-4 text-3xl font-light text-foreground">
              A typed contract: P : X → Y
            </h2>
            <p className="mb-8 max-w-2xl text-lg leading-relaxed text-muted-foreground">
              Fixed input/output schemas. The leaf output fields are the scoring dimensions.
            </p>
            <CodeBlock
              code={`program:
  input:
    text: string
  output:
    invoice_number: string
    date: date
    supplier:
      name: string
      tax_id: string
    subtotal: float
    total: float`}
              language="yaml"
              title="program"
            />
          </div>
        </section>

        {/* The scoring function */}
        <section className="bg-muted/10 px-6 py-24">
          <div className="mx-auto max-w-4xl">
            <span className="section-label mb-4 block">The scoring function</span>
            <h2 className="mb-4 text-3xl font-light text-foreground">
              What "good" means, in numbers
            </h2>
            <p className="mb-8 max-w-2xl text-lg leading-relaxed text-muted-foreground">
              Every output leaf gets one explicit weight. The biggest weight sits on what the business cares about — and every field is reported against a trivial baseline, so a field that beats doing nothing shows it.
            </p>
            <CodeBlock
              code={`scoring:
  weights:
    invoice_number: 1
    date: 1
    supplier:
      name: 1
      tax_id: 0    # validated, reported, but does not move the score
    subtotal: 1
    total: 5       # what the business actually cares about
  aggregator: weighted_mean`}
              language="yaml"
              title="scoring"
            />
          </div>
        </section>

        {/* The AI-system */}
        <section className="px-6 py-24">
          <div className="mx-auto max-w-4xl">
            <span className="section-label mb-4 block">The AI-system</span>
            <h2 className="mb-4 text-3xl font-light text-foreground">
              Any system, one contract
            </h2>
            <p className="mb-8 max-w-2xl text-lg leading-relaxed text-muted-foreground">
              A model, an optimized prompt, a workflow, a tool-using agent — all the same from outside: named fields in, named fields out.
            </p>
            <CodeBlock
              code={`# your own code, bound at the command line
ai-system:
  type: external
  id: invoice-extractor-v7

# or a model — benchy picks a built-in adapter
ai-system:
  type: model
  provider: together               # together | bedrock | openai
  model: Qwen/Qwen3.8-2.4T-A95B
  prompt: ./prompt.md              # optional`}
              language="yaml"
              title="ai-system"
            />
          </div>
        </section>

        {/* Run */}
        <section className="bg-muted/10 px-6 py-24">
          <div className="mx-auto max-w-4xl">
            <span className="section-label mb-4 block">Execution</span>
            <h2 className="mb-4 text-3xl font-light text-foreground">
              Compile once, run from the IR
            </h2>
            <p className="mb-8 max-w-2xl text-lg leading-relaxed text-muted-foreground">
              YAML compiles to a canonical JSON IR; the engine consumes only the IR. Same benchmark, any provider — only <span className="font-mono text-base">ai-system</span> differs.
            </p>
            <CodeBlock
              code={`# run straight from the YAML
benchy run examples/invoices/benchmark.yaml \\
  --adapter examples/invoices/system.py:extractor

# or compile once and run from the IR — identical result
benchy compile benchmark.yaml -o ir.json
benchy run ir.json --adapter system.py:extractor`}
              title="terminal"
            />
            <div className="mt-6">
              <CodeBlock
                code={`{
  "version": "1.0",
  "benchmark_score": 0.8148148148148149,
  "summary": {"examples": 3, "valid": 3, "invalid_outputs": 0, "execution_errors": 0},
  "results": [...]
}`}
                language="json"
                title="result"
              />
            </div>
          </div>
        </section>

        {/* What it refuses */}
        <section className="px-6 py-24">
          <div className="mx-auto max-w-4xl">
            <span className="section-label mb-4 block">No repair, by design</span>
            <h2 className="mb-4 text-3xl font-light text-foreground">
              It accepts, or it rejects
            </h2>
            <p className="mb-8 max-w-2xl text-lg leading-relaxed text-muted-foreground">
              No normalization, no repair, no inferred defaults — one semantic concept, one syntax. A failure keeps <span className="font-mono text-base">null</span> so the diagnosis survives, and counts 0.
            </p>
            <CodeBlock
              code={`{"phase": "compile", "code": "missing_weight", "path": ["supplier", "tax_id"],
 "message": "output leaf has no weight"}`}
              language="json"
              title="structured diagnostic"
            />
          </div>
        </section>

        {/* Architecture */}
        <section id="docs" className="bg-muted/10 px-6 py-24">
          <div className="mx-auto max-w-5xl">
            <span className="section-label mb-4 block text-center">Documentation</span>
            <h2 className="mb-10 text-center text-3xl font-light text-foreground">
              Architecture
            </h2>
            <ArchitectureDiagrams />
          </div>
        </section>

        {/* Footer */}
        <footer className="px-6 py-8">
          <div className="flex items-center justify-center gap-4">
            <a
              href="https://discord.com/invite/VSqAr8BhKV"
              className="text-muted-foreground/60 transition-colors hover:text-primary"
              target="_blank"
              rel="noopener noreferrer"
            >
              <svg className="h-5 w-5" viewBox="0 0 24 24" fill="currentColor">
                <path d="M20.317 4.37a19.791 19.791 0 0 0-4.885-1.515.074.074 0 0 0-.079.037c-.21.375-.444.864-.608 1.25a18.27 18.27 0 0 0-5.487 0 12.64 12.64 0 0 0-.617-1.25.077.077 0 0 0-.079-.037A19.736 19.736 0 0 0 3.677 4.37a.07.07 0 0 0-.032.027C.533 9.046-.32 13.58.099 18.057a.082.082 0 0 0 .031.057 19.9 19.9 0 0 0 5.993 3.03.078.078 0 0 0 .084-.028 14.09 14.09 0 0 0 1.226-1.994.076.076 0 0 0-.041-.106 13.107 13.107 0 0 1-1.872-.892.077.077 0 0 1-.008-.128 10.2 10.2 0 0 0 .372-.292.074.074 0 0 1 .077-.01c3.928 1.793 8.18 1.793 12.062 0a.074.074 0 0 1 .078.01c.12.098.246.198.373.292a.077.077 0 0 1-.006.127 12.299 12.299 0 0 1-1.873.892.077.077 0 0 0-.041.107c.36.698.772 1.362 1.225 1.993a.076.076 0 0 0 .084.028 19.839 19.839 0 0 0 6.002-3.03.077.077 0 0 0 .032-.054c.5-5.177-.838-9.674-3.549-13.66a.061.061 0 0 0-.031-.03zM8.02 15.33c-1.183 0-2.157-1.085-2.157-2.419 0-1.333.956-2.419 2.157-2.419 1.21 0 2.176 1.096 2.157 2.42 0 1.333-.956 2.418-2.157 2.418zm7.975 0c-1.183 0-2.157-1.085-2.157-2.419 0-1.333.955-2.419 2.157-2.419 1.21 0 2.176 1.096 2.157 2.42 0 1.333-.946 2.418-2.157 2.418z"/>
              </svg>
            </a>
            <a href="https://surus.lat" className="text-sm text-muted-foreground/60 transition-colors hover:text-primary">by SURUS</a>
          </div>
        </footer>
      </main>
    </div>
  );
};

export default Index;
