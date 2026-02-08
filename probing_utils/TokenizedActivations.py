import torch
from transformer_lens import HookedTransformer, ActivationCache
from .TokenizedInput import TokenizedInput


class TokenizedActivations:
	tokens: TokenizedInput
	hook_points: list[str]
	activations: Float[Tensor, "batch sequence layer activation"]
	probe_target_mask: Float[Tensor, "batch sequence layer activation"]
	
	def __init__(self, batched_sequences_tokenized: TokenizedInput, model:HookedTransformer, hook_points: list[str] | str, batch_size: int = 1):
		assert batch_size > 0
		self.tokens = batched_sequences_tokenized
		self.hook_points = [hook_points] if isinstance(hook_points,str) else hook_points
		device = next(model.parameters()).device
		names_filter = lambda name: name in self.hook_points
		#token_ids = self.input.tokenized_input.to(device).unsqueeze(0)
		batched_activations = []
		num_prompts = self.tokens.tokenized_input.shape[0]
		for i in range(0, num_prompts, batch_size):
			with torch.no_grad():
				_, cache = model.run_with_cache(
					self.tokens.tokenized_input[i: min(i + batch_size, num_prompts),:].to(device),
					names_filter=names_filter)
				batched_activations.append(
					torch.stack([cache[hook] for hook in self.hook_points], dim=-2))
				del cache
		self.activations = torch.cat(batched_activations, dim=0)
		if self.tokens.probe_target_mask is not None:
			self.probe_target_mask = self.tokens.probe_target_mask.unsqueeze(-1).unsqueeze(-1).expand_as(self.activations)
		else: 
			self.probe_target_mask = self.activations.attention_mask.unsqueeze(-1).unsqueeze(-1).expand_as(self.activations)
