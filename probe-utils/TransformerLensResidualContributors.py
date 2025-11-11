from ProbedModel import *
from transformer_lens import HookedTransformer
import torch as t

class TransformerLensResidualContributors(ProbedModel):
    def __init__(self, model_name:str, activations = "All", device = "mps"):
        self.tl_model = HookedTransformer.from_pretrained(model_name).to(device)
        self.activations = []
        if activations == "All":
            self.activations = [hook_name for hook_name in self.tl_model.hook_dict.keys() 
                                if any(pattern in hook_name for pattern in ['hook_attn_out', 'hook_mlp_out', 'hook_embed'])]
    
    def ActivationNames(self):
        return self.activations
    
    def ActivationShape(self, activation = None):
        return t.shape((len(self.activations),self.tl_model.cfd.d_model))
    
    def Run(self, tokens):
        _, cache = self.tl_model.run_with_cache(tokens, names_filter = self.activations)
        return t.stack(list(cache.values()),dim=-2)
        
	
