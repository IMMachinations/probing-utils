import torch as t


class TrainingRunBuilder:
    def __init__(self):
        self.trainingRun = TrainingRun()
    def build(self):
        return self.trainingRun
