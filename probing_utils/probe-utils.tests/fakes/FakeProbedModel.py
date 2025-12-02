import ProbedModel

import torch as t

class FakeProbedModel(ProbedModel):
	
	def GetActivations(self):
		return ["1","2","3"]

	def ActivationShape(self, activation = None):
		return t.size([10])

	def Run(self,input):
		return None
