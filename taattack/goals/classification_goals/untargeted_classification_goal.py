import numpy as np

from taattack.goals.result import Result
from taattack.goals.status import Status
from .classification_goal import ClassificationGoal


class UntargetedClassificationGoal(ClassificationGoal):
    def __call__(self, workload_list):
        results = []

        probs = self.get_probs(workload_list)
        scores = self.get_scores(workload_list, probs)

        for i in range(len(workload_list)):
            workload = workload_list[i]

            pred = probs[i].argmax()
            workload.orig.queries += 1
            workload.queries = workload.orig.queries
            results.append(Result(
                workload,
                Status.SUCCESSFUL if pred != workload.label else Status.SEARCHING,
                score=scores[i].item(),
            ))

        return results

    def get_scores(self, workload_list, probs=None):
        scores = np.empty(len(workload_list))

        if probs is None:
            probs = self.get_probs(workload_list)

        for i, workload in enumerate(workload_list):
            if not hasattr(workload.orig, 'probs') or workload.orig.probs is None:
                workload.orig.probs = self.get_probs([workload.orig])[0]
            orig_probs = workload.orig.probs
            label = workload.label
            pred = probs[i].argmax()
            scores[i] = (orig_probs[label] - probs[i][label]) + (label != pred) * (probs[i][pred] - orig_probs[pred])

        return scores
