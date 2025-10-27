from datasets import load_dataset

class TextOnlyDataset:
    def __init__(self, hf_title, batch_size):
        self.batch_size = batch_size
        self.hf_dataset = load_dataset(hf_title, split="train", streaming=True)
        self.iterator = self.batch_iter(self.hf_dataset, batch_size)
    def batch_iter(self, stream, bs):
        buf = []
        for ex in stream:
            ids = _tok(ex["text"], return_tensors = "pt", truncation = True, max_length = seq_len, padding = "max_length")["input_ids"][0]
            buf.append(ids)
            if len(buf) == bs:
                yield torch.stack(buf, dim=0); buf.clear()
    def start_epoch(self):
        self.iterator = self.batch_iter(self.hf_dataset, self.batch_size)
    def Next(self):
        return next(self.iterator)
        