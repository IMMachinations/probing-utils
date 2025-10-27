import torch as t
from typing import Optional, Callable, List, Tuple
from sae_lens import HookedSAETransformer

class ActivationDatasetGenerator:
    """Base class for generating activation datasets from text."""
    
    def __init__(
        self, textdataset, model_name: str, hook_names: List[str], inject_layer: int, read_layers: List[int]
    ):
        self.transformer = HookedSAETransformer.from_pretrained_no_processing(model_name)
        self.dataset = textdataset
        self.hook_names = hook_names
        self.inject_layer = inject_layer
        self.read_layers = read_layers
        
    def start_epoch(self):
        self.dataset.start_epoch()
    
    def _extract_activations(self, xb, use_modification: bool = False) -> t.Tensor:
        """Extract activations from specified hooks."""
        raise NotImplementedError("Subclasses must implement _extract_activations")
    
    def Next(self) -> Tuple[t.Tensor, t.Tensor]:
        """
        Generate next batch of (activations, labels).
        Returns:
            X: [2B, L, d_model] - concatenated standard and modified activations
            y: [2B, L] - labels (0 for standard/upstream, 1 for modified downstream)
        """
        xb = self.dataset.Next()
        
        # Get activations without and with modification
        Z_std = self._extract_activations(xb, use_modification=False)  # [B, L, d_model]
        Z_mod = self._extract_activations(xb, use_modification=True)   # [B, L, d_model]
        
        # Concatenate
        X = t.cat([Z_std, Z_mod], dim=0)  # [2B, L, d_model]
        
        # Create labels: 0 for unmodified, 1 for modified downstream layers only
        L = len(self.read_layers)
        layer_mask = t.tensor(
            [1.0 if rl >= self.inject_layer else 0.0 for rl in self.read_layers],
            device=X.device,
            dtype=t.float32
        )  # [L]
        
        y = t.cat([
            t.zeros(Z_std.size(0), L, device=X.device, dtype=t.float32),  # standard: all 0
            layer_mask.unsqueeze(0).expand(Z_mod.size(0), -1),  # modified: 1 if downstream
        ], dim=0)  # [2B, L]
        
        return X, y


class ModifiedActivationDatasetGenerator(ActivationDatasetGenerator):
    def __init__(self, textdataset, model_name: str, hook_names: List[str], inject_layer: int, read_layers: List[int], inject_hook: str, modification_hook: Callable):
        super().__init__(textdataset, model_name, hook_names, inject_layer, read_layers)
        self.inject_hook = inject_hook
        self.modification_hook = modification_hook
    
    @t.no_grad()
    def _extract_activations(self, xb):
        """Extract activations, optionally with modification applied."""
        with self.transformer.hooks(fwd_hooks=[(self.inject_hook, self.modification_hook)]):
            _, cache = self.transformer.run_with_cache(xb, names_filter=self.hook_names, pos_slice=-1, return_cache_object=True, clear_contexts=True)
            
        # Extract and stack activations
        outs = [cache[n].squeeze(1).to(t.float32) for n in self.hook_names]  # [B, d_model] each
        del cache
        return t.stack(outs, dim=1)  # [B, L, d_model]
