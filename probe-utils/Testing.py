import pytest

import ProbeTrainer 
from ProbeTrainer import *

def test_create_trainer():
    trainer = TrainingRunBuilder()
    assert hasattr(trainer, "trainingRun")
    assert trainer.trainingRun == None

def test_trainer_builder():
    trainer = TrainingRunBuilder()
    assert trainer.trainingRun == None
    trainer.BuildRun()
    assert trainer.trainingRun != None
