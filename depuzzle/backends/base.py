from abc import ABC, abstractmethod

from depuzzle.models import BackendInfo


class BaseBackend(ABC):

    @abstractmethod
    def generate(self, prompt):
        pass

    @abstractmethod
    def get_info(self) -> BackendInfo:
        pass
