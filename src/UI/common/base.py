from abc import ABC, abstractmethod

class AppArgProcessor(ABC):
    def __init__(self, arg_value):
        self.value = arg_value
    
    @abstractmethod
    def process(self):
        pass
