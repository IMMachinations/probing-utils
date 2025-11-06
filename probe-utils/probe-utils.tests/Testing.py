import pytest

from TrainingRun import *
#from TrainingRunBuilder import *
from DatasetGenerator import *

def test_create_trainer_builder():
    builder = TrainingRunBuilder()
    assert builder != None

def test_trainer_builder():
    builder = TrainingRunBuilder()
    trainer = builder.build()
    assert trainer != None

def test_activation_generator():
    builder = ActivationDatasetGeneratorBuilder()
    assert builder != None
    generator = builder.build()
    assert generator != None

