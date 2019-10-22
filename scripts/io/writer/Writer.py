from abc import ABC, abstractmethod

class Writer(ABC):
    def __init__(self, path=''):
        self.path = path

    @abstractmethod
    def write(self):
        pass
