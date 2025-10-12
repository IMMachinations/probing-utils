import torch as t
from TrainingRun import *

class TrainingRun:
    def __init__(self):
        self.dataset = None
        self.model = None
        self.init = False
    def Run(self):
        if(not self.init):
            self.initialize();
        for epoch in range(self.num_epochs):
            self.dataset.start_epoch()
            for step in range(self.num_steps):
                self.SingleStep(epoch, step)
        return self.model
    
    def SingleStep(self, epoch, step):
        X, y = self.dataset.Next()

        outputs = self.model(X)
        
        self.optimizer.zero_grad()
        loss = self.loss(outputs, y)
        self.optimizer.step()

        self.log(X,y,loss,step)

        self.eval(step)
        return 
    

