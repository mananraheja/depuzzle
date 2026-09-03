from abc import ABC, abstractmethod

from depuzzle.models import BackendInfo, ExecutionConfig


class BaseBackend(ABC):
    @abstractmethod
    def generate(self, prompt, execution_config: ExecutionConfig | None = None):
        pass

    @abstractmethod
    def get_info(self) -> BackendInfo:
        pass

    @abstractmethod
    def prepare(self):
        """Prepare the model for inference."""
        pass

    @abstractmethod
    def unload(self):
        """Unload the model from memory."""
        pass
