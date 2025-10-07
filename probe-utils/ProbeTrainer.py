import torch as t

class TrainingRunBuilder:
    def __init__(self):
        self.trainingRun = None
    def BuildRun(self):
        self.trainingRun = TrainingRun()
    def Run(self):
        self.trainingRun.Run()

class TrainingRun:
    def __init__(self):
        self.dataset = None
        self.model = None
    def Run(self):
        return self.model
