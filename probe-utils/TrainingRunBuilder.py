import torch as t
from datasets import load_dataset
from TrainingRun import *
from ProbingDataset import *
from StackedLinearProbe import *

class TrainingRunBuilder:
    def __init__(self):
        self.trainingRun = TrainingRun()
        self.probe = None
        self.dataset = None
        self.device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if t.cuda.is_available() else "cpu"))
        print(self.device)
 
    def build(self):
        return self.trainingRun

    def use_dataset(self, dataset : ProbingDataset):
        self.dataset = dataset
        self.trainingRun.dataset = self.dataset
        return self

    def use_probe(self, probe_type : str):
        if(self.dataset == None):
            raise ValueError("Cannot add a probe without a dataset to read probe sizing from.")           
        if(self.dataset.InputShape()[0] != self.dataset.LabelShape()[0]):
            raise ValueError("Dataset activations and labels do not have the same Hook dimension.")
        num_hooks,activation_length = self.dataset.InputShape()
        label_length = self.dataset.LabelShape()[-1]
        if(probe_type == "linear"):
            self.probe =  StackedLinearProbe(num_hooks, activation_length, label_length).to(self.device)
            self.trainingRun.probe = self.probe
        return self

    def use_optimizer(self, optim_type : string):
        if(self.probe == None):
            raise ValueError("Cannot add an optimizer without a probe to optimize")
        if(optim_type == "sgd"):
            self.optimizer = torch.optim.SGD(self.probe.parameters())
            self.trainingRun.optimizer = self.optimizer
        return self
    def use_loss(self, loss_fn : str):
        if(loss_fn == "mse"):
            self.loss = torch.nn.MSELoss()
            self.trainingRun.loss = self.loss
        return self