The goal of this project is to create a fast, easily adaptable library for training linear(or other types) of language model probes.

Towards that we want our top-level interfaces to be as simple as possible, and have the builders take care of the meat of the interaction and connection between components. As such, they should only expose the minimum information required for matching with the other components. Further complications in logic should be handled by subclasses or decorators. 

# Interfaces
## ProbedModel
This interface is for objects that contain the actual model being probed in the training run
Contains the following attributes:
	1. The activations that the model exposes
	2. The shape of the activations that the model exposes
	3. A batched run function that transforms an input into a dictionary(perhaps later some more efficient method) of activations for the labeler

## DatasetGenerator
This interface is for the dataset that the probe is trained on. 
It exposes the following attributes:
	1. The shape of the X input to the model
	2. The shape of the y labels for the model
	3. A batched generator function that exudes paired X and y tensors

# DatasetClasses
## CachedDataset 
This DatasetGenerator is a wrapper around an existing DatasetGenerator that saves the output of the wrapped dataset to disk if it doesn't already exist, and then loads it into memory on subsequent runs to increase training time
