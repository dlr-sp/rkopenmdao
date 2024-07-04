from abc import ABC, abstractmethod


class CheckpointInterface(ABC):
    """Abstract interface for checkpointing implementations."""

    @abstractmethod
    def setup(self, **kwargs):
        pass

    @abstractmethod
    def create_checkpointer(self):
        pass

    @abstractmethod
    def iterate_forward(self, initial_state):
        pass

    @abstractmethod
    def iterate_reverse(self, final_state_perturbation):
        pass
