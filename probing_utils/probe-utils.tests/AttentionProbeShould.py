import pytest
import torch
from transformers import AutoTokenizer
from transformer_lens import HookedTransformer

from probing_utils.TokenizedInput import TokenizedInput
from probing_utils.TokenizedActivations import TokenizedActivations
from probing_utils.AttentionProbe import AttentionProbe


class TestAttentionProbe:
    """Test suite for AttentionProbe class"""

    @pytest.fixture
    def probe_config(self):
        """Configuration for AttentionProbe"""
        return {
            "d_model": 768,
            "n_heads": 4
        }

    @pytest.fixture
    def attention_probe(self, probe_config):
        """Create an AttentionProbe instance"""
        return AttentionProbe(
            d_model=probe_config["d_model"],
            n_heads=probe_config["n_heads"]
        )

    @pytest.fixture
    def mock_activations(self, probe_config):
        """Create mock activations tensor for testing"""
        batch_size = 3
        sequence_length = 10
        d_model = probe_config["d_model"]

        # Create random activations with shape [batch, sequence, activation]
        activations = torch.randn(batch_size, sequence_length, d_model)
        return activations

    def test_initialization(self, attention_probe, probe_config):
        """Test that AttentionProbe initializes correctly"""
        assert hasattr(attention_probe, 'query_probe')
        assert hasattr(attention_probe, 'value_probe')

        # Check parameter shapes
        assert attention_probe.query_probe.shape == (probe_config["n_heads"], probe_config["d_model"])
        assert attention_probe.value_probe.shape == (probe_config["n_heads"], probe_config["d_model"])

        # Check that parameters are trainable
        assert attention_probe.query_probe.requires_grad
        assert attention_probe.value_probe.requires_grad

    def test_forward_pass_output_shape(self, attention_probe, mock_activations):
        """Test that forward pass produces correct output shape"""
        output = attention_probe(mock_activations)

        # Output should be [batch]
        expected_shape = torch.Size([mock_activations.shape[0]])
        assert output.shape == expected_shape

    def test_forward_pass_with_different_batch_sizes(self, probe_config):
        """Test forward pass with various batch sizes"""
        probe = AttentionProbe(d_model=probe_config["d_model"], n_heads=probe_config["n_heads"])

        for batch_size in [1, 2, 5, 10]:
            activations = torch.randn(batch_size, 8, probe_config["d_model"])
            output = probe(activations)

            assert output.shape == torch.Size([batch_size])

    def test_forward_pass_output_range(self, attention_probe, mock_activations):
        """Test that output values are in valid sigmoid range [0, 1]"""
        output = attention_probe(mock_activations)

        # Sigmoid output should be between 0 and 1
        assert torch.all(output >= 0)
        assert torch.all(output <= 1)

    def test_forward_pass_is_differentiable(self, attention_probe, mock_activations):
        """Test that forward pass creates a computational graph for gradients"""
        output = attention_probe(mock_activations)

        # Check that output requires grad
        assert output.requires_grad

        # Verify we can compute gradients
        loss = output.sum()
        loss.backward()

        assert attention_probe.query_probe.grad is not None
        assert attention_probe.value_probe.grad is not None


class TestAttentionProbeIntegration:
    """Integration tests for AttentionProbe with TokenizedActivations"""

    @pytest.fixture(scope="class")
    def tiny_model_and_tokenizer(self):
        """Load a tiny model and tokenizer"""
        model = HookedTransformer.from_pretrained("gpt2-small", device="cpu")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.chat_template = "{% for message in messages %}{{ message['role'] }}: {{ message['content'] }}\n{% endfor %}"
        return model, tokenizer

    def test_forward_pass_with_real_activations(self, tiny_model_and_tokenizer):
        """Test AttentionProbe with real activations from TokenizedActivations"""
        model, tokenizer = tiny_model_and_tokenizer

        # Create prompts
        prompts = [
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"}
            ],
            [
                {"role": "user", "content": "How are you?"},
                {"role": "assistant", "content": "I'm doing well, thank you!"},
                {"role": "user", "content": "That's great to hear"}
            ],
            [
                {"role": "system", "content": "You are a coding assistant."},
                {"role": "user", "content": "Write a hello world"},
                {"role": "assistant", "content": "print('Hello, World!')"}
            ]
        ]

        # Create TokenizedInput
        batch_inputs = TokenizedInput(prompts, tokenizer)

        # Get activations from a single layer
        hook_points = ["blocks.0.hook_resid_post"]

        activations = TokenizedActivations(
            batched_sequences_tokenized=batch_inputs,
            model=model,
            hook_points=hook_points,
            batch_size=1
        )

        # Get d_model from activations
        d_model = activations.activations.shape[-1]
        n_heads = 4

        # Create AttentionProbe
        probe = AttentionProbe(d_model=d_model, n_heads=n_heads)

        # Get activations for a single layer (remove layer dimension)
        # activations.activations shape: [batch, sequence, layer, activation]
        single_layer_activations = activations.activations[:, :, 0, :]

        # Run forward pass
        output = probe(single_layer_activations)

        # Verify output shape
        assert output.shape == torch.Size([3])  # 3 prompts

        # Verify output is in valid range
        assert torch.all(output >= 0)
        assert torch.all(output <= 1)

    def test_forward_pass_with_batched_activations_multiple_layers(self, tiny_model_and_tokenizer):
        """Test AttentionProbe can process activations from multiple sequences"""
        model, tokenizer = tiny_model_and_tokenizer

        # Create a larger batch
        prompts = [
            [
                {"role": "user", "content": "What is machine learning?"},
                {"role": "assistant", "content": "Machine learning is a subset of AI."}
            ],
            [
                {"role": "system", "content": "You are a math tutor."},
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "2+2 equals 4"}
            ],
            [
                {"role": "user", "content": "Tell me a joke"},
                {"role": "assistant", "content": "Why did the chicken cross the road?"},
                {"role": "user", "content": "Why?"}
            ],
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "How does photosynthesis work?"}
            ],
            [
                {"role": "user", "content": "Write Python code for factorial"},
                {"role": "assistant", "content": "def factorial(n): return 1 if n <= 1 else n * factorial(n-1)"}
            ]
        ]

        batch_inputs = TokenizedInput(prompts, tokenizer)

        hook_points = ["blocks.0.hook_resid_post", "blocks.1.hook_resid_post"]

        activations = TokenizedActivations(
            batched_sequences_tokenized=batch_inputs,
            model=model,
            hook_points=hook_points,
            batch_size=2
        )

        d_model = activations.activations.shape[-1]
        probe = AttentionProbe(d_model=d_model, n_heads=8)

        # Test with first layer
        first_layer_activations = activations.activations[:, :, 0, :]
        output = probe(first_layer_activations)

        assert output.shape == torch.Size([5])
        assert torch.all(output >= 0)
        assert torch.all(output <= 1)
