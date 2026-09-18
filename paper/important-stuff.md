# notes

[https://chatgpt.com/g/g-p-6a9e0988a3d48191ab9a49e41d55d936-benchy/c/6a9e09a9-bfa4-83e9-add0-47845eee959e](https://chatgpt.com/g/g-p-6a9e0988a3d48191ab9a49e41d55d936-benchy/c/6a9e09a9-bfa4-83e9-add0-47845eee959e)

[technical-design.md](http://technical-design.md) instead of manifiesto.  
Or technical-paper.md? \`paper\` points to 

The UI may provide defaults. The canonical YAML should make those defaults explicit.

The YAML should be self-sufficient.

Program schema→Data contract​

Every dataset example must conform to the program:

xi∈Xx\_i\\in X yi∗​∈Y

Don't create an abstraction before there is pressure requiring it.?????

scoring function is constrained by the program schema.

data is constrained by program schema.

your draw is correct, i like the dag form

The program schema is therefore the **root semantic object**.  
The data must conform to the program.  
The scoring function must refer only to things defined by the program output schema.

a **program has two schemas with different roles**.  
Program=Input Schema+Output Schema  
Schema=Field Structure+Semantic Types  
`input` and `output` are not part of the field structure. They define the **role** of each schema.  
Within either schema:

* structure says **what fields exist and how they relate**  
* types say **what those fields mean / what values are valid**

And it gives us a nice minimal hierarchy of concepts:

semantic types→schemas→program→benchmark

This thing above is showing dependency/composition, not a definition of schema. 

The definitions remain:  
Schema=Field Structure+Semantic Types\\boxed{ \\text{Schema} \= \\text{Field Structure} \+ \\text{Semantic Types} }

Then:

Program=Input Schema+Output Schema\\boxed{ \\text{Program} \= \\text{Input Schema} \+ \\text{Output Schema} }

Then a benchmark uses the program:

Benchmark=Program+Scoring Function+Data\\boxed{ \\text{Benchmark} \= \\text{Program} \+ \\text{Scoring Function} \+ \\text{Data} }

So the more accurate dependency picture is:

Semantic Types ─┐  
                ├─→ Schema ─┐  
Field Structure ┘           ├─→ Program ─→ Benchmark  
                            │  
                    Input / Output role

Or mathematically:  
(Field Structure,Semantic Types)→Schema(\\text{Field Structure},\\text{Semantic Types}) \\rightarrow \\text{Schema} (Input Schema,Output Schema)→Program(\\text{Input Schema},\\text{Output Schema}) \\rightarrow \\text{Program} (Program,Scoring,Data)→Benchmark

Then B1 has a very strong invariant:  
**Every program has a fixed output schema​**

**Explicit is better than implicit.**  
**explicit is better than implicit. given that, redundancy is not desired.**

**Avoid adding complexity through oversimplification.**  
**The user should need to know as little about Benchy as possible.**  
**The YAML should be self-sufficient.**  
**Nothing about the runtime should leak into benchmark semantics unless necessary.**  
**Expressivity through composition of a very small vocabulary.**  
**The UI edits YAML; agents edit YAML; humans edit YAML. The YAML is the benchmark.**  
**Types express semantic constraints, not storage representation.**  
**Every program has a fixed output schema​. This yields: fixed program schema⇒fixed scoring dimensions⇒fixed weights**  
**fixed output schema⇒fixed evaluation dimensions⇒fixed field weights⇒simple scoring**

**Outputs may be scalar or fixed structured schemas. Variable-length output collections are unsupported.**

items:  
  \- ...

### **2\. I think the cleaner formalization is now**

Not:

Program Schema=field structure+semantic types\\text{Program Schema}=\\text{field structure}+\\text{semantic types}

because a **program has two schemas with different roles**.

Instead:

Program=Input Schema+Output Schema\\boxed{ \\text{Program} \= \\text{Input Schema} \+ \\text{Output Schema} }

and independently:

Schema=Field Structure+Semantic Types​

So:

```
program:
  input:
    image: image

  output:
    invoice_number: string
    date: date
    total: float
```

`input` and `output` are not part of the field structure. They define the **role** of each schema.

# technical-design-doc

\# Benchy B1 — Technical Design

\#\# 1\. Objective / Vision

Benchy is a universal benchmark-definition and execution engine for AI programs.

The unit of evaluation is not a model. It is an \*\*AI system implementing a program\*\*.

A system may be:

\- a model,  
\- a model \+ prompt,  
\- a fine-tuned model,  
\- a workflow,  
\- multiple models,  
\- an agent,  
\- an endpoint encapsulating arbitrary logic.

Benchy does not care how the system works internally. It cares only whether the system satisfies the program contract and how well it performs on the benchmark.

| Math | Representation |  
|---|---|  
| \\(M:X\\rightarrow Y\\) | \`prediction \= system(input)\` |

The long-term goal is a \*\*general semantic language for AI benchmarks\*\* that is simple enough for humans and agents to author directly, while remaining independent of the execution runtime.

The canonical artifact is YAML.

\> \*\*The YAML is the benchmark definition. Everything else is machinery.\*\*

Humans edit YAML.    
Agents edit YAML.    
The UI edits YAML.    
Version control stores YAML.    
Python, Rust, hosted services, or other runtimes consume the same YAML.

\---

\#\# 2\. Design Principles

\#\#\# 2.1 Explicit is better than implicit

The canonical YAML should contain the information required to understand the benchmark without knowledge of hidden Benchy defaults.

Bad:

\`\`\`yaml  
scoring:  
  aggregator: mean  
\`\`\`

Better:

\`\`\`yaml  
scoring:  
  weights:  
    invoice\_number: 1  
    date: 1  
    total: 1  
  aggregator: weighted\_mean  
\`\`\`

The UI may create defaults for the user, but the serialized benchmark should make them explicit.

\---

\#\#\# 2.2 The YAML should be self-sufficient

A reader should be able to inspect the YAML and answer:

1\. What program is being evaluated?  
2\. How is it scored?  
3\. What is the exam?  
4\. What system is taking it?

No knowledge of Benchy internals should be required.

\---

\#\#\# 2.3 Nothing about the runtime should leak into benchmark semantics unless necessary

The benchmark semantic layer must not depend on Python classes, Rust structs, provider SDKs, or object-oriented abstractions.

The engine maps the semantic layer into runtime code.

\`\`\`text  
Benchy YAML  
    ↓  
Benchy Engine  
    ↓  
Runtime / Provider / Endpoint / Workflow  
\`\`\`

\---

\#\#\# 2.4 Expressivity through a very small vocabulary

Benchy should avoid task-specific classes such as \`InvoiceInput\`, \`InvoiceOutput\`, \`TranscriptionResult\`, etc.

The benchmark language should compose a small set of semantic types and field structures.

\> \*\*Types express semantic constraints, not storage representation.\*\*

A \`date\` may serialize as a string, but semantically it is still a date.

\---

\#\#\# 2.5 Avoid complexity through oversimplification

Simplification must not remove information that is necessary to understand the benchmark.

Examples:

\- omitting equal field weights is too implicit,  
\- reducing a classification output to \`string\` loses the closed set of valid labels,  
\- allowing variable output dimensions in B1 would make scoring semantics unclear.

\---

\#\#\# 2.6 B1 has a fixed output schema

This is a core invariant.

\\\[  
\\boxed{  
\\text{fixed output schema}  
\\Rightarrow  
\\text{fixed evaluation dimensions}  
\\Rightarrow  
\\text{fixed field weights}  
\\Rightarrow  
\\text{simple scoring}  
}  
\\\]

Variable-length output collections are deliberately outside B1 because they complicate:

\- field weights,  
\- exact-match semantics,  
\- matching predicted and expected items,  
\- aggregation,  
\- the UI.

\---

\# 3\. Canonical Benchmark Definition

The Benchy UI exposes four structural steps:

\\\[  
\\boxed{  
\\text{Program}  
\+  
\\text{Scoring Function}  
\+  
\\text{Data}  
\+  
\\text{System}  
}  
\\\]

\`\`\`yaml  
program: ...  
scoring: ...  
data: ...  
system: ...  
\`\`\`

These are the four objects the user configures.

\---

\# 4\. Pillar 1 — Program

The program defines \*\*what must be done\*\*.

\\\[  
\\boxed{  
P \= (\\text{Input Schema}, \\text{Output Schema})  
}  
\\\]

or:

\\\[  
P:X\\rightarrow Y  
\\\]

\`\`\`yaml  
program:  
  input: ...  
  output: ...  
\`\`\`

\#\# 4.1 Schema

A schema is:

\\\[  
\\boxed{  
\\text{Schema}  
\=  
\\text{Field Structure}  
\+  
\\text{Semantic Types}  
}  
\\\]

\#\#\# B1 semantic types

\`\`\`text  
string  
int  
float  
bool  
enum  
date  
time  
datetime  
image  
audio  
document  
\`\`\`

Semantic meaning:

| Type | Meaning |  
|---|---|  
| \`string\` | unconstrained text |  
| \`int\` | integer |  
| \`float\` | real-valued number |  
| \`bool\` | \`true\` / \`false\` |  
| \`enum\` | value from a closed categorical set |  
| \`date\` | calendar date |  
| \`time\` | time of day |  
| \`datetime\` | date \+ time |  
| \`image\` | one visual artifact treated as an image |  
| \`audio\` | audio artifact |  
| \`document\` | document artifact that may contain pages, text, images, and layout |

An \`enum\` defines a finite domain:

\\\[  
Y=\\{v\_1,v\_2,\\ldots,v\_k\\}  
\\\]

\`\`\`yaml  
output:  
  enum: \[positive, neutral, negative\]  
\`\`\`

\#\# 4.2 Examples

\#\#\# Transcription

| Math | YAML |  
|---|---|  
| \\(P:\\text{audio}\\rightarrow\\text{string}\\) | \`input: audio\` → \`output: string\` |

\`\`\`yaml  
program:  
  input: audio  
  output: string  
\`\`\`

\#\#\# Classification

| Math | YAML |  
|---|---|  
| \\(P:\\text{string}\\rightarrow\\{\\text{positive},\\text{neutral},\\text{negative}\\}\\) | \`output.enum\` |

\`\`\`yaml  
program:  
  input: string  
  output:  
    enum: \[positive, neutral, negative\]  
\`\`\`

\#\#\# Invoice extraction

| Math | YAML |  
|---|---|  
| \\(P:\\text{image}\\rightarrow Y\\) | fixed structured output |

\`\`\`yaml  
program:  
  input: image  
  output:  
    invoice\_number: string  
    date: date  
    supplier: string  
    subtotal: float  
    total: float  
\`\`\`

Nested fixed structures are allowed:

\`\`\`yaml  
program:  
  input: document  
  output:  
    supplier:  
      name: string  
      tax\_id: string  
    date: date  
    total: float  
\`\`\`

The set of output fields remains fixed.

\---

\# 5\. Pillar 2 — Scoring Function

The scoring function defines \*\*what good performance means\*\*.

For structured outputs:

\> \*\*A scoring function specifies the relative importance of output fields and the rule by which field-level correctness is aggregated into program-level performance.\*\*

B1 exposes two user-defined parts:

\\\[  
\\boxed{  
\\text{Scoring Function}  
\=  
\\text{Field Weights}  
\+  
\\text{Aggregator}  
}  
\\\]

\`\`\`yaml  
scoring:  
  weights:  
    invoice\_number: 1  
    date: 1  
    supplier: 1  
    subtotal: 1  
    total: 5  
  aggregator: weighted\_mean  
\`\`\`

\#\# 5.1 Field weights

Weights define \*\*importance\*\*, not correctness.

\\\[  
w=(w\_1,\\ldots,w\_n)  
\\\]

\`\`\`yaml  
weights:  
  invoice\_number: 1  
  date: 1  
  supplier: 1  
  subtotal: 1  
  total: 5  
\`\`\`

A weight of \`5\` means that field contributes five times as much as a field with weight \`1\`.

All weights should be explicit in canonical YAML.

\#\# 5.2 Aggregator

The aggregator determines how field-level correctness becomes one instance score.

\\\[  
A(c,w)\\rightarrow s  
\\\]

where \\(c\\) is the runtime correctness vector.

Weighted mean:

\\\[  
\\boxed{  
A(c,w)=  
\\frac{\\sum\_i w\_i c\_i}  
{\\sum\_i w\_i}  
}  
\\\]

\`\`\`python  
score \= weighted\_mean(field\_scores, weights)  
\`\`\`

Other possible aggregators include:

\`\`\`text  
min  
max  
median  
sum  
weighted\_mean  
\`\`\`

Example: with binary field scores,

\\\[  
\\min(1,1,0)=0  
\\\]

so \`min\` implements all-or-nothing correctness.

\#\# 5.3 Scoring validity is constrained by the program schema

A scoring function may only reference fields defined by the output schema.

\`\`\`yaml  
program:  
  output:  
    supplier: string  
    total: float

scoring:  
  weights:  
    supplier: 1  
    total: 5  
\`\`\`

Valid.

\`\`\`yaml  
scoring:  
  weights:  
    supplier: 1  
    tax: 5  
\`\`\`

Invalid if \`tax\` is not in the output schema.

Formally:

\\\[  
\\boxed{  
\\text{Program Schema}  
\\rightarrow  
\\text{Scoring-Function Validation}  
}  
\\\]

\---

\# 6\. Pillar 3 — Data

The data is the \*\*exam\*\*.

Each example contains:

\\\[  
(x\_i,y\_i^\*)  
\\\]

where:

\- \\(x\_i\\) is the exam question,  
\- \\(y\_i^\*\\) is the known correct answer.

The complete dataset is:

\\\[  
D=\\{(x\_i,y\_i^\*)\\}\_{i=1}^{N}  
\\\]

The benchmark YAML only points to the dataset:

\`\`\`yaml  
data:  
  path: ./data/invoices.jsonl  
\`\`\`

The program schema constrains the dataset:

\\\[  
x\_i\\in X  
\\\]

\\\[  
y\_i^\*\\in Y  
\\\]

Therefore:

\\\[  
\\boxed{  
\\text{Program Schema}  
\\rightarrow  
\\text{Data Validation}  
}  
\\\]

The dependency structure is:

\`\`\`text  
                 Program Schema  
                 /            \\  
                /              \\  
               ↓                ↓  
        Data Validation   Scoring Validation  
\`\`\`

The scoring function does \*\*not\*\* define the shape of the data. Both the data and scoring function are independently constrained by the program schema.

\#\# 6.1 Synthetic data

Synthetic benchmark generation should also obey the same contract.

\\\[  
G(P,S,\\text{context})\\rightarrow D  
\\\]

but every generated example must satisfy:

\\\[  
D\\models P  
\\\]

The benchmark definition therefore provides the semantic constraints required to validate generated data.

\---

\# 7\. Pillar 4 — System

The system is the \*\*exam taker\*\*.

\\\[  
M:X\\rightarrow Y  
\\\]

Benchy is AI-system-agnostic.

Two B1 configuration modes are useful.

\#\# 7.1 Model configuration

For the common case:

\`\`\`yaml  
system:  
  type: model  
  provider: openai  
  model: \<model\>  
  prompt: ./prompts/invoice.md  
  parameters:  
    temperature: 0  
\`\`\`

The important concepts are:

\`\`\`text  
provider  
model  
prompt  
inference parameters  
\`\`\`

The exact provider schema may evolve.

\#\# 7.2 Endpoint configuration

For arbitrary systems:

\`\`\`yaml  
system:  
  type: endpoint  
  endpoint: https://example.com/extract  
\`\`\`

The endpoint may encapsulate:

\- one model,  
\- multiple models,  
\- deterministic code,  
\- retrieval,  
\- agents,  
\- validators,  
\- retries,  
\- any other implementation.

From Benchy's perspective, both modes satisfy the same contract:

\\\[  
M:X\\rightarrow Y  
\\\]

\---

\# 8\. Benchmark Execution Layer

The four pillars above define the benchmark run.

The user does not configure runtime field scores or correctness vectors.

For each example:

\\\[  
(x\_i,y\_i^\*)  
\\\]

the system produces:

\\\[  
\\hat y\_i=M(x\_i)  
\\\]

\`\`\`python  
prediction \= system(example.input)  
expected \= example.expected  
\`\`\`

\#\# 8.1 Field-level correctness

B1 uses exact match.

For every scored output field \\(j\\):

\\\[  
c\_j=  
\\mathbf 1\[  
\\hat y\_j=y\_j^\*  
\]  
\\\]

\`\`\`python  
field\_score \= int(predicted\_field \== expected\_field)  
\`\`\`

All field scores form the correctness vector:

\\\[  
c=(c\_1,\\ldots,c\_n)  
\\\]

Example:

\`\`\`text  
expected   \= \[A123, 2026-09-07, 500.0\]  
prediction \= \[A123, 2026-09-07, 450.0\]

c \= \[1, 1, 0\]  
\`\`\`

A \*\*field score\*\* is one \\(c\_j\\).

The \*\*correctness vector\*\* is the collection of all field scores.

These are runtime outputs, not benchmark-definition inputs.

\#\# 8.2 Instance score

The scoring function combines the runtime correctness vector with the predefined weights:

\\\[  
s\_i=A(c\_i,w)  
\\\]

Example:

\\\[  
c=(1,1,0)  
\\\]

\\\[  
w=(1,1,5)  
\\\]

Weighted mean:

\\\[  
s=  
\\frac{1(1)+1(1)+5(0)}  
{1+1+5}  
\=  
\\frac{2}{7}  
\\\]

\`\`\`python  
score \= weighted\_mean(  
    field\_scores=\[1, 1, 0\],  
    weights=\[1, 1, 5\],  
)  
\`\`\`

\#\# 8.3 Benchmark score

For \\(N\\) exam examples:

\\\[  
s\_1,\\ldots,s\_N  
\\\]

B1 benchmark score:

\\\[  
\\boxed{  
B(M)=  
\\frac{1}{N}  
\\sum\_{i=1}^{N}s\_i  
}  
\\\]

\`\`\`python  
benchmark\_score \= mean(instance\_scores)  
\`\`\`

Complete execution:

\`\`\`text  
input  
  ↓  
system  
  ↓  
prediction  
  ↓  
compare with expected output  
  ↓  
field scores / correctness vector  
  ↓  
weights \+ aggregator  
  ↓  
instance score  
  ↓  
aggregate across exam  
  ↓  
benchmark score  
\`\`\`

\---

\# 9\. Minimal Semantic Object Model

This is conceptual, not a requirement to implement classes.

\`\`\`text  
Program  
  input\_schema  
  output\_schema

ScoringFunction  
  weights  
  aggregator

Data  
  path

System  
  configuration  
\`\`\`

Canonical YAML:

\`\`\`yaml  
program:  
  input: image  
  output:  
    invoice\_number: string  
    date: date  
    supplier: string  
    subtotal: float  
    total: float

scoring:  
  weights:  
    invoice\_number: 1  
    date: 1  
    supplier: 1  
    subtotal: 1  
    total: 5  
  aggregator: weighted\_mean

data:  
  path: ./data/invoices.jsonl

system:  
  type: model  
  provider: openai  
  model: \<model\>  
  prompt: ./prompts/invoice.md  
  parameters:  
    temperature: 0  
\`\`\`

This YAML is the canonical benchmark definition.

The UI should expose the same structure as four steps:

\`\`\`text  
1\. Program  
2\. Scoring Function  
3\. Data  
4\. System  
\`\`\`

Changing the UI changes the YAML.

Changing the YAML changes the benchmark.

\---

\# 10\. B1 Limitations / Open Work

\#\# 10.1 Field correctness

B1 supports exact match only:

\\\[  
c\_j=  
\\mathbf 1\[  
\\hat y\_j=y\_j^\*  
\]  
\\\]

This works well for many structured tasks:

\- extraction,  
\- classification,  
\- IDs,  
\- booleans,  
\- exact categorical outputs.

It is insufficient for tasks where semantically equivalent outputs may differ lexically:

\- question answering,  
\- summarization,  
\- translation,  
\- free-form generation.

Future versions will need a generalized field evaluator:

\\\[  
C\_j(\\hat y\_j,y\_j^\*)\\rightarrow\[0,1\]  
\\\]

Possible future evaluators include:

\`\`\`text  
normalized exact match  
numeric tolerance  
set match  
semantic similarity  
LLM-as-judge  
task-specific evaluators  
\`\`\`

These should extend the scoring layer without changing the core benchmark ontology.

\#\# 10.2 Variable-length outputs

B1 does not support variable-length output collections.

The reason is structural, not accidental:

\\\[  
\\text{variable output dimensions}  
\\Rightarrow  
\\text{variable evaluation dimensions}  
\\\]

which complicates:

\`\`\`text  
weights  
matching  
exact-match semantics  
aggregation  
UI representation  
\`\`\`

B1 therefore keeps the invariant:

\\\[  
\\boxed{  
\\text{Every program has a fixed output schema.}  
}  
\\\]

\---

\# 11\. Core Summary

\\\[  
\\boxed{  
\\text{Program}  
\=  
\\text{Input Schema}  
\+  
\\text{Output Schema}  
}  
\\\]

\\\[  
\\boxed{  
\\text{Schema}  
\=  
\\text{Field Structure}  
\+  
\\text{Semantic Types}  
}  
\\\]

\\\[  
\\boxed{  
\\text{Scoring Function}  
\=  
\\text{Field Weights}  
\+  
\\text{Aggregator}  
}  
\\\]

\\\[  
\\boxed{  
\\text{Data}  
\=  
\\text{Exam Questions}  
\+  
\\text{Known Correct Answers}  
}  
\\\]

\\\[  
\\boxed{  
\\text{System}  
\=  
\\text{Implementation Taking the Exam}  
}  
\\\]

The program schema is the root semantic constraint:

\`\`\`text  
                 Program Schema  
                 /            \\  
                /              \\  
               ↓                ↓  
            Data             Scoring  
\`\`\`

And execution is:

\\\[  
\\boxed{  
x\_i  
\\xrightarrow{M}  
\\hat y\_i  
\\xrightarrow{\\text{compare with }y\_i^\*}  
c\_i  
\\xrightarrow{S}  
s\_i  
}  
\\\]

followed by:

\\\[  
\\boxed{  
(s\_1,\\ldots,s\_N)  
\\rightarrow  
B(M)  
}  
\\\]

Benchy's core design goal is therefore:

\> \*\*A small, explicit, self-sufficient semantic language for defining and running universal AI-program benchmarks.\*\*

# things i love

In order of chats

ok i really like this: """ For structured outputs, a scoring function specifies the relative importance of output fields and the rule by which field-level correctness is aggregated into program-level performance. """

this is really good: ""\*\*"\*\*The important design constraint should be:

> **Nothing about the runtime should leak into the benchmark semantics unless absolutely necessary."""**

> 

> 

and this is really good as well: """That means the UI edits YAML.

Agents edit YAML.

Humans edit YAML.

Version control stores YAML.

A Python implementation consumes the same YAML.

A Rust implementation consumes YAML.

A hosted Benchy service consumes the same YAML.

The YAML is the benchmark.

Everything else is machinery.

And I think the principle you're reaching for could be stated very simply:

```
\boxed{\text{Expressivity through composition of a very small vocabulary.}}
```

Not lots of classes for lots of tasks.

Just:

**types → schemas → programs → scoring → benchmarks.**

That is much closer to something I would want to build a long-lived system around."""

i think your definition of image vs document is on point: """ `image` vs `document`  
 I would make the distinction semantic:

* `image` \= one visual artifact treated as an image.  
* `document` \= a document artifact that may contain multiple pages, text, images, layout, etc. """

"""The scoring function instead has to be **valid with respect to the output schema**.

For example:

```

```

```
scoring:
  weights:
    total: 5
```

is invalid if `total` does not exist in the output schema.

But that is:

Program schema→Scoring-function validation"""

And i really like this: 

“””  
And yes, your DAG is now very clean:

Program Schema\\boxed{\\text{Program Schema}}

branches into two constraints:

Program Schema↙↘DataScoring Function\\begin{array}{ccc} & \\text{Program Schema} & \\\\ \\swarrow && \\searrow \\\\ \\text{Data} && \\text{Scoring Function} \\end{array}

Meaning:

D⊨PD \\models P

The data must conform to the program.

And:

S⊨PS \\models P

The scoring function must refer only to things defined by the program output schema.

The program schema is therefore the **root semantic object**.

“””

i really like this: """The principle is: **types express semantic constraints, not storage representation."""**

in the age of ai and human-ai collaboration, semantics is very important.

i really like this:

"""2. I think the cleaner formalization is now  
 Not:

```
\text{Program Schema}=\text{field structure}+\text{semantic types}
```

because a **program has two schemas with different roles**.

Instead:

```
\boxed{
\text{Program}
=
\text{Input Schema}
+
\text{Output Schema}
}
```

and independently:

Schema=Field Structure+Semantic Types​""" i think its on point.

And i really like this:   
“””  
And on this:

semantic types→schemas→program→benchmark\\text{semantic types}\\rightarrow\\text{schemas}\\rightarrow\\text{program}\\rightarrow\\text{benchmark}

I was showing **dependency/composition**, not giving another definition of schema.

The definitions remain:

Schema=Field Structure+Semantic Types\\boxed{ \\text{Schema} \= \\text{Field Structure} \+ \\text{Semantic Types} }

Then:

Program=Input Schema+Output Schema\\boxed{ \\text{Program} \= \\text{Input Schema} \+ \\text{Output Schema} }

Then a benchmark uses the program:

Benchmark=Program+Scoring Function+Data\\boxed{ \\text{Benchmark} \= \\text{Program} \+ \\text{Scoring Function} \+ \\text{Data} }

So the more accurate dependency picture is:

```
Semantic Types ─┐
                ├─→ Schema ─┐
Field Structure ┘           ├─→ Program ─→ Benchmark
                            │
                    Input / Output role
```

Or mathematically:

(Field Structure,Semantic Types)→Schema(\\text{Field Structure},\\text{Semantic Types}) \\rightarrow \\text{Schema} (Input Schema,Output Schema)→Program(\\text{Input Schema},\\text{Output Schema}) \\rightarrow \\text{Program} (Program,Scoring,Data)→Benchmark(\\text{Program},\\text{Scoring},\\text{Data}) \\rightarrow \\text{Benchmark}

That is probably the cleaner formulation  
“”””

This is good: 

“””

Then B1 has a very strong invariant:

`Every program has a fixed output schema`

“””

This is grea as well:   
“””

that restriction, gives you:

`fixed program schema⇒fixed scoring dimensions⇒fixed weights\boxed{ \text{fixed program schema} \Rightarrow \text{fixed scoring dimensions} \Rightarrow \text{fixed weights} }`

That is beautifully aligned with the scoring-function abstraction we've built.

Then later, when Benchy needs variable-length outputs, you can introduce them deliberately as a new capability:

\`collections\\text{collections}\`

which will require its own evaluation semantics: matching, ordering, missing elements, extra elements, etc.

…

The clean chain is:

`fixed output schema⇒fixed evaluation dimensions⇒fixed field weights⇒simple scoring​`

“””

This is good as well:   
“””

For B1:

> **Outputs may be scalar or fixed structured schemas. Variable-length output collections are unsupported.**

“””

Task should be explicit, but not free-form. It should come from a controlled task ontology, and each task should constrain the class of valid program schemas.

## **Task vs Program**

I would define:

\`\`\`Task=family of programs\\boxed{\\text{Task} \= \\text{family of programs}}\`\`\`

and:

\`\`\`Program=specific typed contract​\`\`\`

extract:X→structured fixed schema

classify:X→enum

Task⊃many valid Programs

Benchy can then validate:

P⊨T\\boxed{ P \\models T }

> Does this program satisfy the declared task?

                   Task

                     ↓

              Program Schema

               /          \\

              ↓            ↓

            Data         Scoring

Task→valid Program space 

Program→valid Data space 

Program→valid Scoring space

task=what operation is performed​   
domain=what distribution/world it is performed on  
language=what linguistic distribution it is performed on

Task constrains the **program**.

Domain and language characterize primarily the **data distribution**.

And it solves the core problem:

> `task` is neither an arbitrary label nor something blindly inferred from I/O types.

It is a **controlled semantic class with structural constraints**.

I think that's probably the missing layer between your task-oriented ontology and the Benchy program abstraction.

When Benchy loads the YAML, it should do semantic validation before running anything:  
parse YAML

  ↓

validate syntax

  ↓

validate task ↔ program compatibility

  ↓

validate program ↔ scoring compatibility

  ↓

validate program ↔ data compatibility

  ↓

compile benchmark

On P models T  
“””  
The useful reading of

P⊨TP \\models T

is closer to:

> **PP satisfies TT**  
>  **PP conforms to the constraints expressed by TT**

For example, imagine TT describes a valid class of programs:

T={input has field ‘text‘,output has field ‘sentiment‘,sentiment∈{positive, negative}}T \= \\{ \\text{input has field \`text\`}, \\text{output has field \`sentiment\`}, \\text{sentiment}\\in\\{\\text{positive, negative}\\} \\}

and PP is a concrete program/schema.

Then

P⊨TP \\models T

means:

> **P is one of the things allowed by T.**

Visually, you can think of TT as defining a **set of valid possibilities**:

Models(T)={P1,P2,P3,…}\\mathrm{Models}(T) \= \\{P\_1,P\_2,P\_3,\\ldots\\}

and

P⊨T⟺P∈Models(T)P \\models T \\quad\\Longleftrightarrow\\quad P \\in \\mathrm{Models}(T)

That is probably the **“contained within / constrained by” intuition** you're getting.

But there's an important directional distinction:

P⊨T\\boxed{P \\models T}

does **not** literally mean:

P⊆TP \\subseteq T

Instead, it means:

P obeys the constraints specified by T\\boxed{\\text{P obeys the constraints specified by T}}

So if, in Benchy, **TT is the specification / type / theory defining the space**, then I would mentally read:

P⊨TP \\models T

as:

> **“P is a valid inhabitant of T.”**

That framing is much closer to what we're doing with Benchy: a definition constrains a space of possible concrete objects, and a particular object either **satisfies** that definition or doesn't.

“””

“””  
So I would separate:

Benchmark Definition\\boxed{\\text{Benchmark Definition}}

from:

Ontology Definition\\boxed{\\text{Ontology Definition}}

The benchmark says:

> I am a `classify` benchmark.

The ontology says:

> `classify` means programs whose output is an enum.

And the Benchy compiler checks that the two agree.

I think the clean architecture is:

```
Task Ontology ──────┐
                    ↓
Benchmark YAML → Benchy Compiler → Executable Benchmark
```

The ontology constrains the benchmark definition; the benchmark definition constrains execution.

“””

“””  
The underlying theory can still say:

> Benchmarks are classified using the SURUS task-oriented AI ontology `/task/domain/language/`.

But the YAML doesn't need to expose the word **ontology**.

I would distinguish:

* **Ontology** → the canonical external system defining valid tasks/domains/languages and their relationships.  
* **Benchmark classification** → the particular `task`, `domain`, and `language` assigned to this benchmark.

And in the UI, don't say “ontology” or “taxonomy” at all. Just ask:

```
Task
Domain
Language
```

That better satisfies the principle that **the user should need to understand as little machinery as possible**.

“””

Task registry

“””  
tasks:

  extract:

    input: any

    output: structured

  classify:

    input: any

    output: enum

  transcribe:

    input: audio

    output: string

  translate:

    input: string

    output: string

  summarize:

    input: \[string, document\]

    output: string  
“””

# collections (or dynamic output schema)

that restriction.

It gives you:

fixed program schema⇒fixed scoring dimensions⇒fixed weights\\boxed{ \\text{fixed program schema} \\Rightarrow \\text{fixed scoring dimensions} \\Rightarrow \\text{fixed weights} }

That is beautifully aligned with the scoring-function abstraction we've built.

Then later, when Benchy needs variable-length outputs, you can introduce them deliberately as a new capability:

collections\\text{collections}

which will require its own evaluation semantics: matching, ordering, missing elements, extra elements, etc.

