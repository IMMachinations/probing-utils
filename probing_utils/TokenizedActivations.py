#from transformers import PreTrainedTokenizerBase
from transformer_lens import HookedTransformer, ActivationCache
from .LogitTarget import LogitTarget
from .TokenizedInput import TokenizedInput


class TokenizedActivations:
    input: TokenizedInput
    activations: ActivationCache
    def __init__(self, tokens_in:TokenizedInput, model:HookedTransformer, hook_point_filter):
        self.input = tokens_in
        _, self.activations = model.run_with_cache(self.input.input_string,hook_point_filter)
        