import torch as t
from datasets import load_dataset
from TrainingRun import *


class TrainingRunBuilder:
    def __init__(self):
        self.trainingRun = TrainingRun()
        self.model = None
        self.dataset = None
        self.device = ("cuda" if t.cuda.is_available() else "cpu")
        print(self.device)
 
    def build(self):
        return self.trainingRun

    def use_dataset(self, dataset):
        self.trainingRun.dataset = dataset
        return self

    def use_model(self, model):
        self.trainingRun.model = model
        return self
        

