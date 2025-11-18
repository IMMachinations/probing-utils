from TransformerLensResidualContributors import *
from ProbingDataset import *
import datasets 
import numpy as np
import torch


class SpecificLabeledHFDataset(ProbingDataset):
    def __init__(self):
        self.dataset = datasets.load_dataset('ucberkeley-dlab/measuring-hate-speech', 'default', columns = ['text','sentiment','respect','insult','humiliate','status','dehumanize','violence','genocide', 'attack_defend','hatespeech'])
        self.tlmodel = TransformerLensResidualContributors("gpt2")
        self.device = self.tlmodel.on_device()
        self.NUMERIC_COLS = ['sentiment','respect','insult','humiliate','status','dehumanize','violence','genocide', 'attack_defend','hatespeech']
        def collate_fn(batch):
            numeric_features = torch.stack([
            torch.tensor([item[col] for col in self.NUMERIC_COLS], dtype=torch.float32) for item in batch])
            return {'text':[item['text'] for item in batch], 'numeric_features': numeric_features}
        
        self.dataloader = torch.utils.data.DataLoader(self.dataset['train'], collate_fn=collate_fn)
        self.set_iter()
        
    def InputShape(self, activation = None):
        return self.tlmodel.ActivationShape()

    def LabelShape(self):
        return torch.Size([len(self.tlmodel.ActivationNames()), len(self.NUMERIC_COLS)])
        
    def __len__(self):
        return len(self.dataloader)

    def set_iter(self):
        self.iterable = iter(self.dataloader)
    def __iter__(self):
        for x in self.iterable:
            text, label = x['text'], x['numeric_features'].to(self.device)
            activations = self.tlmodel.Run(text).to(self.device)
            activation_shape = list(activations.shape)
            activation_shape[-2:] = self.LabelShape()
            for _ in range(activations.ndim - label.ndim):
                label = label.unsqueeze(0)
            label = label.expand(activation_shape)
            yield activations.detach(), label.detach()
        
        