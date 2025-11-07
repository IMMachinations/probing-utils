import HFLabeledDataset
import FakeProbedModel
import pytest
import torch as t

def InitializeFailOnEmptyString():
	with pytest.raises(ValueError, match="name is not recognized"):
		dataset = HFLabeledDataset(name = "", model = FakeProbedModel())

def LoadFromExistingDataset():		
	hf_ds = load_dataset("dataset_name", split="train[:10]")
	dataset = HFLabeledDataset(model = FakeProbedModel(), hf_ds)
	assert dataset.ActivationShape() == t.shape([3,10]) 
