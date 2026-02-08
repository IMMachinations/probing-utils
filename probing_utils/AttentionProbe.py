from torch.nn import Module, Parameter
from torch import randn, einsum, Tensor, sigmoid
from torch.nn.functional import softmax
from jaxtyping import Float
from typing import Optional

class AttentionProbe(Module):
	def __init__(self, n_layers:int, d_model:int, n_heads:int):
		super(AttentionProbe,self).__init__()
		self.query_probe = Parameter(randn(n_layers, n_heads, d_model))
		self.value_probe = Parameter(randn(n_layers, n_heads, d_model))
	def forward(self, activations: Float[Tensor, "batch sequence layer activation"], mask: Optional[Float[Tensor, "batch sequence"]] = None):
		queries = einsum('bplx,lhx->bplh', activations, self.query_probe)
		values = einsum('bplx,lhx->bplh', activations, self.value_probe)

		if mask is not None:
			mask_expanded = mask.unsqueeze(-1).unsqueeze(-1)  # [batch, sequence, 1, 1]
			queries = queries.masked_fill(mask_expanded == 0, float('-inf'))

		attention = softmax(queries, dim=1)
		output = einsum('bplh,bplh->bl', attention, values)
		return sigmoid(output)
