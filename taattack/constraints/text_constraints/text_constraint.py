from abc import ABC, abstractmethod


class TextConstraint(ABC):
    def __call__(self, workload):
        return self._check(workload)

    @abstractmethod
    def _check(self, workload):
        return NotImplemented()

    def filter(self, workload_list):
        return list(filter(self._check, workload_list))
