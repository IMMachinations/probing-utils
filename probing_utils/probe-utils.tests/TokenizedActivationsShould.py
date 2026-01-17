import pytest
import torch
from transformers import AutoTokenizer
from transformer_lens import HookedTransformer

from probing_utils.TokenizedInput import TokenizedInput
from probing_utils.TokenizedActivations import TokenizedActivations


class TestTokenizedActivationsIntegration:
    """Integration tests for TokenizedActivations with TokenizedInput"""

    @pytest.fixture(scope="class")
    def tiny_model_and_tokenizer(self):
        """Load a tiny model and tokenizer from transformer_lens"""
        model = HookedTransformer.from_pretrained("gpt2-small", device="cpu")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        # Add a simple chat template for GPT-2
        tokenizer.chat_template = "{% for message in messages %}{{ message['role'] }}: {{ message['content'] }}\n{% endfor %}"
        return model, tokenizer

    def test_captures_activations_from_batch_tokenized_inputs(self, tiny_model_and_tokenizer):
        """Test that TokenizedActivations captures activations from TokenizedInput"""
        model, tokenizer = tiny_model_and_tokenizer

        # Create a small batch of conversational prompts
        prompts = [
            [
                {"role": "user", "content": "Hello"}
            ],
            [
                {"role": "user", "content": "How are you?"}
            ],
            [
                {"role": "assistant", "content": "I am fine"}
            ]
        ]

        # Create TokenizedInput
        batch_inputs = TokenizedInput(prompts, tokenizer)

        # Specify a couple of hook points
        hook_points = ["blocks.0.hook_resid_post", "blocks.1.hook_resid_post"]

        # Create TokenizedActivations with batch_size=1
        activations = TokenizedActivations(
            batched_sequences_tokenized=batch_inputs,
            model=model,
            hook_points=hook_points,
            batch_size=1
        )

        # Verify activations were captured
        assert hasattr(activations, 'activations')
        assert isinstance(activations.activations, torch.Tensor)

        # Verify activations have the right number of dimensions
        assert activations.activations.ndim == 4  # batch, sequence, layer, activation

        # Verify we captured activations for all prompts
        assert activations.activations.shape[0] == 3

        # Verify we captured the right number of layers (hook points)
        assert activations.activations.shape[2] == len(hook_points)

        # Verify activations are not all zeros
        assert not torch.all(activations.activations == 0)

        # Verify the tokens were stored correctly
        assert activations.tokens == batch_inputs
        assert activations.hook_points == hook_points
