from abc import ABC, abstractmethod

class ProbedModel(ABC):
	
	@abstractmethod
	def ActivationNames(self):
		pass

	@abstractmethod
	def ActivationShape(self, activation = None):
		pass
	
	@abstractmethod
	def Run(self):
		pass
	
