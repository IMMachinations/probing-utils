from transformers import PreTrainedTokenizerBase
from .LogitTarget import LogitTarget

class TokenizedInput:
    input_string: str
    tokenized_input: list[int]
    tokenized_string_input: list[str]
    tokenized_probe_targets: list[LogitTarget]
    def __init__(self, input_string:str, tokenizer):
        self.input_string = input_string
        self.tokenized_string_input = tokenizer.tokenize(input_string)
        self.tokenized_input = tokenizer.convert_tokens_to_ids(self.tokenized_string_input)
    