from abc import ABC, abstractmethod

class Reader(ABC):
    def __init__(self, path=''):
        self.path = path

    @abstractmethod
    def read(self):
        pass
