import torch
from transformers import PreTrainedTokenizerBase
from .LogitTarget import LogitTarget

class UnbatchedTokenizedInput:
    input_string: str
    tokenized_input: torch.Tensor
    tokenized_string_input: list[str]
    tokenized_probe_targets: torch.Tensor

    def __init__(self, input_string:str, tokenizer):
        self.input_string = input_string
        self.tokenized_string_input = tokenizer.tokenize(input_string)
        token_ids = tokenizer.convert_tokens_to_ids(self.tokenized_string_input)
        self.tokenized_input = torch.tensor(token_ids, dtype=torch.long)
        self.tokenized_probe_targets = torch.tensor([], dtype=torch.long)

    def set_probe_targets_from_list(self, targets: list[LogitTarget], max_targets_per_token=10):
        """
        Convert list of LogitTarget objects to tensor.
        Returns shape (seq_len, max_targets_per_token) where:
        - -1 indicates no target (either None or padding)
        - Other values are the target indices
        """
        target_matrix = []
        for target in targets:
            if target.value is None:
                row = [-1] * max_targets_per_token
            else:
                row = target.value + [-1] * (max_targets_per_token - len(target.value))
            target_matrix.append(row)
        self.tokenized_probe_targets = torch.tensor(target_matrix, dtype=torch.long)
    