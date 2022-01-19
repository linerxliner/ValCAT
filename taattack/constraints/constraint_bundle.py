from .pre_constraints import PreConstraint
from .text_constraints import TextConstraint


class ConstraintBundle:
    def __init__(self, constraints):
        self._pre_constraints = []
        self._text_constraints = []

        for c in constraints:
            if isinstance(c, PreConstraint):
                self._pre_constraints.append(c)
            elif isinstance(c, TextConstraint):
                self._text_constraints.append(c)

    def pre_filter(self, indices, workload):
        if not self._pre_constraints:
            return indices

        for c in self._pre_constraints:
            indices = c.filter(indices, workload)

        return indices

    def text_filter(self, workload_list):
        if not self._text_constraints:
            return workload_list

        for c in self._text_constraints:
            workload_list = c.filter(workload_list)

        return workload_list
