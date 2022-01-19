from taattack.utils import get_use_sim
from .text_constraint import TextConstraint


class MinUseSim(TextConstraint):
    def __init__(self, threshold=0.8):
        self._threshold = threshold

    def _check(self, workload):
        workload.sim = get_use_sim(workload.orig.text, workload.text)
        r = workload.sim >= self._threshold

        if not r:
            workload.failed_reason = f'USE similarity ({workload.sim}) less than {self._threshold}'

        return r
