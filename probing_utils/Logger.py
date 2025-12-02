from abc import ABC, abstractmethod

class Logger(ABC):
    @abstractmethod
    def log(self, X, y, probeTest, loss, step):
        pass

    @abstractmethod
    def newRun(self):
        pass
    