import torch as t
from datasets import load_dataset

class TrainingRunBuilder:
    def __init__(self):
        self.trainingRun = TrainingRun()
        self.model = None
        self.dataset = None
        self.device = ("cuda" if t.cuda.is_available() else "cpu")
        print(self.device)

    def verifyComponents(self):
        if self.model == None:
            return False
        if self.dataset == None:
            return False
        if not hasattr(self.dataset, 'next'):
            return False
        if not callable(getattr(obj, 'my_method')):
    
    def 
        
    def build(self):
        return self.trainingRun

    def use_dataset(self,hf_name, batch_size):
        self.trainingRun.dataset = DatasetWrapper(hf_name, batch_size)



class DatasetWrapper:
    
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

