import torch
from jaxtyping import Float
from torch import Tensor
from .ProbeTargets import ProbeTargets
from .TokenizedInput import TokenizedInput
from .TokenizedActivations import TokenizedActivations

class BinaryProbeTargets(ProbeTargets):
	labels: list[bool]

	def __init__(self, labels: list[bool], corresponding_inputs: TokenizedInput):
		super().__init__(labels, corresponding_inputs)

	def prompt_level_target(self, activations: TokenizedActivations) -> Float[Tensor, "batch sequence"]:
		batch, sequence = activations.activations.shape[0], activations.activations.shape[1]
		return torch.tensor(self.labels, dtype=torch.float32).unsqueeze(1).expand(batch, sequence)
