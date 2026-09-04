 We have to rethink how we are going to make this tool work because the main objective is to To be
 a tool to expose normal people to making a benchmark for their AI, and our anthology is that of
task-oriented AI. Our tool does that. It's an abstraction that allows you to do a benchmark on a
given task against any AI system, be it a single model, workflow, or agent. The engineering is
there, the engineering for our anthology is there, but we still need to develop the interface to
this and that interface is not great now. It makes you do all these configurations for Benchy to
run a task and it's sort of impenetrable for someone who's not the active developer or knows this
tool a lot. So let's think through together what is needed, how is this approach going, and how is
this path going to be? What I'm thinking is that a user first has to define the benchmark and how
do you define the benchmark? One way to do it is by thinking through tasks, like: "Okay this AI
here is doing an extraction so if it's doing an extraction, what is the input and what is the
output?" There you can define your benchmark. I mean it's this task; it has to do this. This is how
 that function looks: it has these inputs and these outputs and it has to behave like this. Then
comes the question of how are we going to measure success? That's the scoring function. After
defining the inputs and outputs on our benchmark we have to define the scoring function: say how
are we going to measure how well the system works?
There are many different scoring functions but there are some that are common, like:
- give a point if the output is correct given the input and a zero if it's not
- if you have to do a task, you break it into subtasks and you give a point for each subtask that
is performed accurately
For example if you have to extract ten fields, you give one point for each of the fields that you
extract correctly. And after the user has defined the benchmark (that is, what is the task and what
 are the inputs and the outputs) and he has defined how the scoring function is going to be, he has
 to run this benchmark. To run this benchmark he or she needs data. If they have data, Benchy
should be able to input it and use it as the data of the benchmark. Most likely the user is going
to need to synthesize data because when you define the task as a function with these inputs and
these outputs, maybe you have data but it doesn't have that form. You have to sort of synthesize it
 from existing data, that is, formatting it and getting what is needed. In the case when there's no
 data, just a description of how the task is, how it's inputs and outputs look like, how it
behaves, then Benchy should be able to generate that data, of course with an AI model . And then
after all of that one has to decide which model or workflow or agent, which AI system is one going
to benchmark this against? One has to make decisions about which are the models, which are the
parameters, who's the provider of these models, or are you going to run this locally? There's a lot
 of engineering that has to go into that and that's what Benchy has now but we don't care about
that. We care about all the other things. So please think this through and help me go through this
to understand how we should now build the second layer of Benchy, which is the interaction layer,
the functionality layer, since the engineering layers are already built
