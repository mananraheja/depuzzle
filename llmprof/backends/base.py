from abc import ABC, abstractmethod

from llmprof.models import BackendInfo


class BaseBackend(ABC):

    @abstractmethod
    def generate(self, prompt):
        pass

    @abstractmethod
    def get_info(self) -> BackendInfo:
        pass
