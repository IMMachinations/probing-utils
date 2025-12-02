from abc import ABC, abstractmethod
import torch as t

class CachedProbingDataset(ProbingDataset):
	def __init__(self, dataset: ProbingDataset, filename : str):
		self.dataset = dataset
		self.filepath = filename
        self.size = 0
        self.at = 0
            
	def InputShape(self):
		return self.dataset.InputShape()
	def LabelShape(self):
		return self.dataset.LabelShape()
	def Next(self):
        if(self.at >= self.size):
            newElement = self.dataset.Next()
        filename = self.filepath + str(self.at) + ".pt"
        self.at += 1 
        return t.load()
