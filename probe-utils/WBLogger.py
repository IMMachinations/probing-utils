from Logger import *
import wandb

class WBLogger(Logger):
    def __init__(self, wbProjectName: str):
        self.project = wbProjectName
        self.run = None
        
    def newRun(self):
        self.run = wandb.init(project = self.project)

    def log(self, X, y, probeEval, loss, step):
        pass