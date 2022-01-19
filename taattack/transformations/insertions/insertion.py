from abc import abstractmethod

from taattack.transformations.transformation import Transformation


class Insertion(Transformation):
    @abstractmethod
    def _get_transformed_workload_list(self, *args, **kwargs):
        raise NotImplemented()
