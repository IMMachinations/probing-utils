import torch
from transformers import PreTrainedTokenizerBase
from jaxtyping import Int, Bool, jaxtyped
from torch import Tensor

class TokenizedInput:
	prompts: list[list[dict[str,str]]]
	tokenized_input = Int[Tensor, "batch sequence"]
	probe_target_mask = Bool[Tensor, "batch sequence"]
	tokenizer: PreTrainedTokenizerBase
	def __init__(self, prompts: list[dict[str,str]], tokenizer: PreTrainedTokenizerBase):
		self.tokenizer = tokenizer
		self.prompts = prompts
		tokenized_prompt_list = []
		for prompt in prompts:
			for message in prompt:
				assert message.keys() == {"role":"", "content":""}.keys()
		
		outputs = self.tokenizer.apply_chat_template(prompts, tokenize=True, padding=True, return_tensors="pt",return_dict = True)		
		
		self.tokenized_input = outputs["input_ids"]
		self.attention_mask = outputs["attention_mask"]

	def _mask_detect_all(self):
		self.probe_target_mask = self.attention_mask
		return self.probe_target_mask
	

