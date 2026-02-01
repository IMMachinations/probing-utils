from torch.nn import Module, Parameter
from torch import randn, einsum, Tensor, sigmoid
from torch.nn.functional import softmax
from jaxtyping import Float
from typing import Optional

class AttentionProbe(Module):
	def __init__(self, d_model:int, n_heads:int):
		super(AttentionProbe,self).__init__()
		self.query_probe = Parameter(randn(n_heads, d_model))
		self.value_probe = Parameter(randn(n_heads, d_model))
	def forward(self, activations: Float[Tensor, "batch sequence activation"], mask: Optional[Float[Tensor, "batch sequence"]] = None):
		queries = einsum('bpx,hx->bhp', activations, self.query_probe)
		values = einsum('bpx,hx->bhp', activations, self.value_probe)

		if mask is not None:
			mask_expanded = mask.unsqueeze(1)  # [batch, 1, sequence]
			queries = queries.masked_fill(mask_expanded == 0, float('-inf'))

		attention = softmax(queries, dim=-1)
		output = einsum('bhp,bhp->b', attention, values)
		return sigmoid(output)
