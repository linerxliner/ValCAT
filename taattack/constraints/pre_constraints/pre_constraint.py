from abc import ABC, abstractmethod


class PreConstraint(ABC):
    @abstractmethod
    def filter(self, ids, workload):
        raise NotImplemented()
