from abc import abstractmethod
from scipy.special import softmax

from taattack.goals.goal import Goal


class ClassificationGoal(Goal):
    def get_probs(self, workload_list):
        probs = softmax(self.victim(workload_list), axis=1)

        for i in range(len(workload_list)):
            workload_list[i].probs = probs[i]

        return probs

    @abstractmethod
    def get_scores(self, workload_list, prob=None):
        raise NotImplementedError
