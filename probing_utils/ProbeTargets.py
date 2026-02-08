import torch
from abc import ABC, abstractmethod
from jaxtyping import Float
from torch import Tensor
from .TokenizedInput import TokenizedInput
from .TokenizedActivations import TokenizedActivations

class ProbeTargets(ABC):
	labels: list

	def __init__(self, labels: list, corresponding_inputs: TokenizedInput):
		if len(labels) != corresponding_inputs.tokenized_input.shape[0]:
			raise ValueError("Number of labels must match number of input batches")
		self.labels = labels

	@abstractmethod
	def prompt_level_target(self, activations: TokenizedActivations) -> Float[Tensor, "batch sequence"]:
		pass
