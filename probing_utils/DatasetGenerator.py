import torch as t

class DatasetGenerator:
    def Run(self):
        return 0

class ActivationDatasetGeneratorBuilder:
    def __init__(self):
        self.numberGeneratedExamples = 0
    def build(self):
        return DatasetGenerator()
