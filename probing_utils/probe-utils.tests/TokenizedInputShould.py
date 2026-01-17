import pytest
import torch
from unittest.mock import Mock, MagicMock
from transformers import AutoTokenizer

from probing_utils.TokenizedInput import TokenizedInput


class TestTokenizedInput:
    """Test suite for TokenizedInput class"""

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer for testing"""
        tokenizer = Mock()
        tokenizer.apply_chat_template = MagicMock(return_value={
            "input_ids": torch.tensor([[1, 2, 3, 4, 0], [5, 6, 7, 0, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 0, 0]])
        })
        return tokenizer

    @pytest.fixture
    def valid_prompts(self):
        """Create valid prompt structure for testing"""
        return [
            [
                {"role": "user", "content": "Hello, how are you?"},
                {"role": "assistant", "content": "I'm doing well, thank you!"}
            ],
            [
                {"role": "user", "content": "What is the weather like?"}
            ]
        ]

    @pytest.fixture
    def single_prompt(self):
        """Create a single prompt for testing"""
        return [
            [
                {"role": "user", "content": "Test message"}
            ]
        ]

    def test_initialization_with_valid_prompts(self, valid_prompts, mock_tokenizer):
        """Test that TokenizedInput initializes correctly with valid prompts"""
        batch = TokenizedInput(valid_prompts, mock_tokenizer)

        assert batch.prompts == valid_prompts
        assert batch.tokenizer == mock_tokenizer
        assert batch.tokenized_input is not None
        assert batch.attention_mask is not None

    def test_tokenizer_called_correctly(self, valid_prompts, mock_tokenizer):
        """Test that the tokenizer is called with correct parameters"""
        batch = TokenizedInput(valid_prompts, mock_tokenizer)

        mock_tokenizer.apply_chat_template.assert_called_once_with(
            valid_prompts,
            tokenize=True,
            padding=True,
            return_tensors="pt",
            return_dict=True
        )

    def test_tokenized_input_shape(self, valid_prompts, mock_tokenizer):
        """Test that tokenized input has correct shape"""
        batch = TokenizedInput(valid_prompts, mock_tokenizer)

        assert batch.tokenized_input.shape == torch.Size([2, 5])
        assert isinstance(batch.tokenized_input, torch.Tensor)

    def test_attention_mask_shape(self, valid_prompts, mock_tokenizer):
        """Test that attention mask has correct shape"""
        batch = TokenizedInput(valid_prompts, mock_tokenizer)

        assert batch.attention_mask.shape == torch.Size([2, 5])
        assert isinstance(batch.attention_mask, torch.Tensor)

    def test_invalid_prompt_structure_missing_role(self, mock_tokenizer):
        """Test that initialization fails with invalid prompt structure (missing 'role')"""
        invalid_prompts = [
            [
                {"content": "Hello, how are you?"}
            ]
        ]

        with pytest.raises(AssertionError):
            TokenizedInput(invalid_prompts, mock_tokenizer)

    def test_invalid_prompt_structure_missing_content(self, mock_tokenizer):
        """Test that initialization fails with invalid prompt structure (missing 'content')"""
        invalid_prompts = [
            [
                {"role": "user"}
            ]
        ]

        with pytest.raises(AssertionError):
            TokenizedInput(invalid_prompts, mock_tokenizer)

    def test_invalid_prompt_structure_extra_keys(self, mock_tokenizer):
        """Test that initialization fails with invalid prompt structure (extra keys)"""
        invalid_prompts = [
            [
                {"role": "user", "content": "Hello", "extra": "field"}
            ]
        ]

        with pytest.raises(AssertionError):
            TokenizedInput(invalid_prompts, mock_tokenizer)

    def test_mask_detect_all(self, valid_prompts, mock_tokenizer):
        """Test _mask_detect_all returns the attention mask"""
        batch = TokenizedInput(valid_prompts, mock_tokenizer)
        masks = batch._mask_detect_all()

        assert masks.shape == batch.tokenized_input.shape
        assert torch.equal(masks, batch.attention_mask)

    def test_mask_detect_all_returns_attribute(self, valid_prompts, mock_tokenizer):
        """Test _mask_detect_all sets and returns probe_target_mask attribute"""
        batch = TokenizedInput(valid_prompts, mock_tokenizer)
        masks = batch._mask_detect_all()

        assert hasattr(batch, 'probe_target_mask')
        assert torch.equal(batch.probe_target_mask, masks)

    def test_single_prompt_initialization(self, single_prompt, mock_tokenizer):
        """Test initialization with a single prompt"""
        mock_tokenizer.apply_chat_template = MagicMock(return_value={
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]])
        })

        batch = TokenizedInput(single_prompt, mock_tokenizer)

        assert batch.tokenized_input.shape[0] == 1
        assert batch.attention_mask.shape[0] == 1

    def test_empty_prompts_list(self, mock_tokenizer):
        """Test initialization with empty prompts list"""
        mock_tokenizer.apply_chat_template = MagicMock(return_value={
            "input_ids": torch.tensor([]).reshape(0, 0),
            "attention_mask": torch.tensor([]).reshape(0, 0)
        })

        batch = TokenizedInput([], mock_tokenizer)

        assert batch.prompts == []
        assert batch.tokenized_input.numel() == 0

    def test_multiple_messages_per_prompt(self, mock_tokenizer):
        """Test handling multiple messages in a single prompt"""
        prompts = [
            [
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "How are you?"}
            ]
        ]

        mock_tokenizer.apply_chat_template = MagicMock(return_value={
            "input_ids": torch.tensor([[1, 2, 3, 4, 5, 6]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1]])
        })

        batch = TokenizedInput(prompts, mock_tokenizer)

        assert batch.prompts == prompts
        assert batch.tokenized_input is not None


class TestTokenizedInputIntegration:
    """Integration tests using real HuggingFace tokenizers"""

    @pytest.fixture(scope="class")
    def gpt2_tokenizer(self):
        """Load TinyLlama tokenizer from HuggingFace (has built-in chat template)"""
        tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def test_real_tokenizer_with_multiple_conversations(self, gpt2_tokenizer):
        """Test with real GPT-2 tokenizer and multiple diverse conversations"""
        prompts = [
            [
                {"role": "system", "content": "You are a helpful assistant that provides concise answers."},
                {"role": "user", "content": "What is the capital of France?"},
                {"role": "assistant", "content": "The capital of France is Paris."}
            ],
            [
                {"role": "user", "content": "How does photosynthesis work?"},
                {"role": "assistant", "content": "Photosynthesis is the process by which plants convert light energy into chemical energy."}
            ],
            [
                {"role": "system", "content": "You are a coding assistant."},
                {"role": "user", "content": "Write a hello world in Python"},
                {"role": "assistant", "content": "print('Hello, World!')"}
            ]
        ]

        batch = TokenizedInput(prompts, gpt2_tokenizer)

        # Verify basic properties
        assert batch.prompts == prompts
        assert batch.tokenizer == gpt2_tokenizer
        assert isinstance(batch.tokenized_input, torch.Tensor)
        assert isinstance(batch.attention_mask, torch.Tensor)

        # Verify batch dimension matches number of prompts
        assert batch.tokenized_input.shape[0] == 3
        assert batch.attention_mask.shape[0] == 3

        # Verify shapes match between input and mask
        assert batch.tokenized_input.shape == batch.attention_mask.shape

        # Verify tokenized output is not empty
        assert batch.tokenized_input.numel() > 0

        # Verify all token IDs are valid (non-negative)
        assert torch.all(batch.tokenized_input >= 0)

        # Verify attention mask contains only 0s and 1s
        assert torch.all((batch.attention_mask == 0) | (batch.attention_mask == 1))

    def test_real_tokenizer_mask_detect_all(self, gpt2_tokenizer):
        """Test _mask_detect_all with real tokenizer"""
        prompts = [
            [
                {"role": "user", "content": "Test message one"}
            ],
            [
                {"role": "user", "content": "Test message two with more tokens"}
            ]
        ]

        batch = TokenizedInput(prompts, gpt2_tokenizer)
        masks = batch._mask_detect_all()

        # Verify mask shape matches input
        assert masks.shape == batch.tokenized_input.shape

        # Verify mask is a tensor
        assert isinstance(masks, torch.Tensor)

        # Verify mask equals attention mask
        assert torch.equal(masks, batch.attention_mask)

        # Verify probe_target_mask attribute is set
        assert hasattr(batch, 'probe_target_mask')
        assert torch.equal(batch.probe_target_mask, batch.attention_mask)

    def test_real_tokenizer_padding_behavior(self, gpt2_tokenizer):
        """Test that padding works correctly with different length prompts"""
        prompts = [
            [
                {"role": "user", "content": "Short"}
            ],
            [
                {"role": "user", "content": "This is a much longer message that should result in more tokens"}
            ]
        ]

        batch = TokenizedInput(prompts, gpt2_tokenizer)

        # Both sequences should have same length due to padding
        assert batch.tokenized_input.shape[0] == 2
        seq_len = batch.tokenized_input.shape[1]

        # First sequence should have some padding (attention mask has 0s)
        first_attention = batch.attention_mask[0]
        assert torch.sum(first_attention == 0) > 0, "First sequence should have padding"

        # Second sequence should have fewer or no padding tokens
        second_attention = batch.attention_mask[1]
        first_padding_count = torch.sum(first_attention == 0)
        second_padding_count = torch.sum(second_attention == 0)
        assert first_padding_count >= second_padding_count
