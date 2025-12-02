import torch as t
from tqdm import tqdm

class TrainingRun:
    def __init__(self):
        self.dataset = None
        self.probe = None
        self.init = False
        self.num_epochs = 1
        self.optimizer = None
        self.logger = None
        self.eval = None
        self.loss = None
    def Run(self):
        if(not self.init):
            #self.initialize();
            pass
        for epoch in range(self.num_epochs):
            #self.dataset.start_epoch()
            pbar = tqdm(enumerate(self.dataset), total=len(self.dataset))
            for step, (activation, label) in pbar:
                self.SingleStep(activation, label, step, epoch)
        return self.model
    
    def SingleStep(self, X, y, step, epoch):

        self.optimizer.zero_grad()
        outputs = self.probe(X)
        
        loss = self.loss(outputs, y)
        
        self.optimizer.step()
        
        probeEval = None
        if(self.eval is not None):
            probeEval = self.evaluator.evaluate(step)
        if(self.logger is not None):
            self.logger.log(X, y, probeEval, loss, step)

        return 
    

    def initialize(self):
        pass