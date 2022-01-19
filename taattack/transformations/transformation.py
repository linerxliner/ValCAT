from abc import ABC, abstractmethod


class Transformation(ABC):
    def __call__(self, *args, **kwargs):
        return self._get_transformed_workload_list(*args, **kwargs)

    @abstractmethod
    def _get_transformed_workload_list(self, *args, **kwargs):
        raise NotImplemented()
