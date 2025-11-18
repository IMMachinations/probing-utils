from abc import ABC, abstractmethod
from collections.abc import Iterable

class ProbingDataset(ABC, Iterable):
    @abstractmethod
    def InputShape(self):
        pass
    
    @abstractmethod
    def LabelShape(self):
        pass
    
    @abstractmethod
    def __len__(self):
        pass
    
    @abstractmethod
    def __iter__(self):
        # Subclasses return an iterator here
        pass