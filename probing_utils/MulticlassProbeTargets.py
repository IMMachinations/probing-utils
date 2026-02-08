import torch
from jaxtyping import Float
from torch import Tensor
from .ProbeTargets import ProbeTargets
from .TokenizedInput import TokenizedInput
from .TokenizedActivations import TokenizedActivations

class MulticlassProbeTargets(ProbeTargets):
	k_classes: int
	labels: list[int]

	def __init__(self, labels: list[int], k_classes:int,  corresponding_inputs: TokenizedInput):
		super().__init__(labels, corresponding_inputs)
		self.k_classes = k_classes
		for label in labels:
			if label < 0 or label >= k_classes:
				raise ValueError("Class index out of range")

	def prompt_level_target(self, activations: TokenizedActivations) -> Float[Tensor, "batch sequence k_classes"]:
		batch, sequence = activations.activations.shape[0], activations.activations.shape[1]
		targets = torch.zeros((batch, sequence, self.k_classes), dtype=torch.float32)
		label_tensor = torch.tensor(self.labels).view(batch, 1, 1).expand(batch, sequence, 1)
		targets.scatter_(2, label_tensor, 1.0)
		return targets
