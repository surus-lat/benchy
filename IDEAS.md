el scoring function se puede derivar del output schema! luego se tunean weights o pasa a binario. 


system core ideas

put full power into going to the bare metal of what
we need. create a bare-metal-golem that guards the p
rocess, making sure we continue pushing to the bare
metal, without being afraid of breaking things. if n
othing is broken in a push cycle to bare metal, we a
re not pushing hard enough

We are pushing.

We are redesigning Benchy in another session and we want to keep pushing to the bare metal of what we need to accomplish the vision of Benchy. We will have many cycles of pushing to the bare metal, knowing that if we didn't break something in one of the push cycles then we're not pushing hard enough. It would be good to try to find that bare metal global minima, to say, in a theoretical way, using theory somehow. As in, OK, what do we need? A benchmark. What do we need for the benchmark? We need to know what the benchmark is for, for example, is an extraction benchmark or what it is that is actually an abstraction of the underlying thing that is describing the program to me. How is the program? Define the program space somehow.

And then once you have the program definition and the program space, you need to somehow define how the exam is going to be for the program and how it is going to be scored. This is sort of like the discrete version of gradient descent where we have a scoring function, which is also a loss function that tells us if one thing is better than another. Then we have instances of that test, that is, you have the data that you've run through the model when you're evaluating it to make it do the inference, like saying, "OK, do the exam."

For example, if we just have to define a program, we can do it in the most pure signal way: the input is this, like a document in PDF, and the output is a JSON extraction, a list of extractive values. For example, then we simply need a scoring function to define how it is going to be scored: if it's a binary one, I'm going to give you one point for each one; if it's a rubric, I have three ones correct and two incorrect. Whatever is...

And then you need the data that represents instances of that. The data is making the AI take the exam. To infer here, it's like saying, "Write on this page," but the thing that is confusing and/or cloudy is that it's not like you do one exam. You do many exams, as many as data items you have. Another way to say it is that an exam has like a hundred pages, say, that you have to go through and on each of the pages you make an inference and then it's scored somehow. You sort of get a point estimate of how well you did because how well you did is a distribution. Everything is a distribution, but we want a point estimate, be it the mean or the max or whatever it is.

And then we have to think how we are going to represent the thing to be evaluated because at the end you can say, Okay it's actually an API. I'm going to evaluate this API but it's also tricky because if everything is an API maybe we have to find another concept here or way to see it somehow. We have to have to make an exam passer or exam writer or exam finisher to take the exam to make an inference. This can be a single model, can be many models working together, can be like one model with an optimized prompt.

I'm not sure what we're doing here. Maybe it's the compiler. Maybe we should think this is the compiler. The compiler is what is going to run all your program into to make the test. The compiler is the one that makes the inference. That's right, that's right. Yeah we got it. We got it. So we need to expose the compiler and to expose the compiler I think this is what it's going to be, somehow hard. There are maybe many systems that are trying to make it work here because on the one side we have to control the output of this compiler to make it match the input scheme. The output scheme of the compiler has to match the input scheme of the exam that has to be taken.

We have to say, "Hey if you're going to take the exam, these are the things that you have to know how to do." It's multiple choice and you have to do this and this and this or it's this one: you have to do this and this and this. In computer language it's this output schema that has too much to do with the input schema of the next process.øHere you can start thinking on things like: if I want to serve a model I can use BLLM but BLLM is only for some types of models. If you want to do something similar, like it's still AI but it's more like old-school machine learning, like an EOLA model for example, you cannot use BLLM.

There's a raw way of making all the models you want to infer. Instead of thinking of models we just think about explicit programs and implicit programs, or written programs and learned programs. We would see very clearly that all the explicit programs and written programs have a compiler of their language but there's no one compiler of all the learned models.

We have this sort of a slew of engines to make them do inference, like PLM, LAMA-CPP, G-Lang, I think it's another one, and then all the old ones. We have PyTorch. I don't know how the fuck do you serve? Do you make inference on a workflow? That's another thing. When you get into the workflow level, say, are you gonna have these three models do something in sequence to make this program? There's the question of how you compile all of this into something. Maybe we should expose a compiler part that sort of tries to run the learned program no matter if it was just an old machine learning model or a neural network or a foundation model or whatever it is. And then if you have a workflow, say, "I have this task that is extract structured data from images." If you say, "Okay my input-output schema, the thing that describes the program that solves this task, which is economically relevant, takes in an image and outputs a JSON for example," maybe that's not the full solution because you cannot put a single model that solves this problem. So you end up doing a workflow where you have first a segmentation model that will segment the different sections of the document and then you'll do an extraction on all of those different segments and sort of append everything together at the end. Maybe when you're appending that you're actually giving it to another model saying, "Hey this is all the extracted parts we got from this, all the segments that are taken from this full document. Please give me the final extracted data from that document."

This is more complex. I mean you have a program that is a composition of programs so one should ask, "How do I compile? How do I have a compiler that works on any learned program and in any composition of learned programs?"

So all of this that I'm saying about the compiler or the API could be represented another way, saying this is an AI API for example. You put whatever you want behind the AI API and the API has to have an output schema that's compliant with the input schema. This is more like a simpler way of seeing it and saying what it does. It's an API, an AI API. That means that it's a point of contact with an AI so you can know how you can trade either information or value or give orders or whatever it is. This is the protocol.

The other one, the one about the compiler, maybe at the end it's explaining what's the underlying thing and giving the right engineering framework. We have these two things at the same time so we could even call it AI API but I think that at the low level we have to think of this as a compiler, a general compiler for learned programs. Be it like a single learned program or a composition of learned programs 

Okay but just to sort of recap, all the things that I'm ranting about are this API and compiler and all these problems. We can take it as a big complexity to tackle in the future but now just call it okay, this is the compiler, and we'll deal with that in the future. Maybe think what's the best design for that or just progressively add support for new models, more likely textures, and whatever it is. That's a whole universe on itself.

If we leave it as a compiler, we'll have this other thing. The compiler being like the exam taker that does the inference, that is, writes in the exam the solution over and over or as many times as data points then 
We can think of this layer of abstraction: what is the bare metal? Leave the bare metal of serving any type of learning program on the side because that's maybe an unsolved problem for the next five years and probably someone else thought about this. I think Mojo from Chris Lattner is trying to do this but that's fine. Let's go again.
In the recap we have:
- the compiler, which is the exam taker
- the inference
- the AI API
- the program that we're trying to find or test, like an instance of thought
You want to make it go through the test, which has not only how you take it, like you have to reply to this multiple choice for example, but also how it's going to be scored. You try to get a thorough analysis. Maybe it's a binary outcome, maybe it's one point per each of the fields, maybe you have something weighted. We have to describe how this is going to be scored because this is going to express what is correct and what is more important in a business case. You have a hierarchy of importance on different things. For example for extraction data maybe there's one field that is critical and the other ones are nice to have.
The way in which you codify all these business criteria and the systems is through a scoring function. That's what allows you to control it. These are the three pillars:
- We need to easily define the task, like the program that we're searching for or in other ways, like what is this exam about. This is a structure extraction in this case.
- What is the way this exam is scored?
- A compiler that makes any AI be able to take the exam through an AI API.




------------- some code ideas ------

1. The system is the argument, not a constructor field. June had Benchmark(task, scoring, data, system). But the vision's headline feature is exporting a benchmark as "a new loss function" for prompt-optimizers — and a loss function's free variable is the thing being optimized. So:

Benchmark = Task + Data + Scoring        # the exam
await bench.run(system)                  # grade a candidate
bench.as_loss()                          # (System) -> float

as_loss is now three lines instead of a bolt-on, and reusing one exam across many systems — the actual daily use — is the native path.







