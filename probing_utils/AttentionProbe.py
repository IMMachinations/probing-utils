from torch.nn import Module, Parameter
from torch import randn, einsum
from torch.nn.functional import softmax

class AttentionProbe(Module):
	def __init__(self, d_model:int, n_heads:int):
		super(AttentionProbe,self).__init__()
		self.query_probe = Parameter(randn(n_heads, d_model))
		self.value_probe = Parameter(randn(n_heads, d_model))
	def forward(self, activations: Float[Tensor, "batch sequence activation"]):
		queries = einsum('bpx,hx->bhp', activations, self.query_probe)
		values = einsum('bph,hx->bhp', activations, self.value_probe)
		attention = softmax(queries, dim=-1)
		output = einsum('bhp,bhp->b', attention, values)
		return sigmpoid(output)
