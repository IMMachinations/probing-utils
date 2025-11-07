from abc import ABC, abstractmethod

class CachedProbingDataset(ProbingDataset):
	def __init__(self, dataset: ProbingDataset, filename : str):
		self.dataset = dataset
		self.filename = filename
	def InputShape(self):
		return self.dataset.InputShape()
	
	def LabelShape(self):
		return self.dataset.LabelShape()
	
	def Next(self):
		return self.dataset.Next()
	
