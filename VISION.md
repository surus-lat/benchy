This is what my vision for benchy is, and what we need for our company's work and thus what we infer other people need.

Most (if not all) benchmark frameworks have 2 key principles:
- they are model-centric, that is, the main primitiver is the ai model, and everything is focused on evaluating a raw model.
- they focus on exposing the popular benchmarks in a developer friendly way, handling engineering problems that come up when you want a framework to run any model against any already created global benchmark.


Benchy has 2 very different key principles: 
- it's ai-system (model, node, workflow, agent) agnostic, that is, every ai system is treated as an ai-program, and the focus is on evaluating a given ai-program, not on evaluating a model, or a node (model + optimized_prompt), or a workflow (composed models) or an agent (one or more models with access to tools and a while loop). the main primitive is the ai-system, not the ai model. 

- its focused on CREATING new custom benchmarks, not on running already existing benchmarks. This is because we see the benchmark as the first step in ai development and the bridge between ai capability and business problems to be solved. when one wants to solve a business problem with ai, one has to turn that problem into a benchmark, and the use that benchmark to create and optimize the best ai system possible. Thus the focus is on how do we create a benchmark that correctly represents the business problem and what really matters for the business. Thus the focus is on easily defining things like the input/output schema (the task we want to perform, the program that will solve the business problem), the scoring function (how we measure what good means in this scenario), the target ai-system (what ai system are we benchmarking), and the data of the benchmark. 
And since the benchmark is the first step to optimizing an ai-system, benchy has to have useful features like exporting the benchmark in a way it can be used as a "new loss function" for software 3.0 optimizations like prompt-optimizers.
And the engineering helpfulness here is on handling all the different ai-system configurations under the hood so one can choose an ai-system and all the things to run it under the hood are handled (using the right framework to run different model architectures, exposing any ai-system as an endpoint or other easy way to evaluate it; supporting local inference and/or remote inference like cloud ai providers, etc)


And the modules of benchy must follow this view of the benchmarking and ai development world. So there should be a module to define the task, one for the scoring function, one for the ai-system, and one for the data. The engineering must go hand in hand with how an ai developer has to think about benchmarks, and the step by step flow to create a new one. Engineering must not be a problem, but an aid to this view of the ai world. 



And keep in mind that this is our whole ontology for ai: 
/<task?>/<domain?>/<language?>

an ai system is a program that performs a task, thus that's the root level, then a sublevel is the domain, then the third sublevel is the language. 
