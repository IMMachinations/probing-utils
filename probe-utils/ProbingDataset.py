from abc import ABC, abstractmethod

class ProbingDataset(ABC):
	@abstractmethod
	def InputShape(self):
		pass
	@abstractmethod
	def LabelShape(self):
		pass
	@abstractmethod
	def Next(self):
		pass
	
