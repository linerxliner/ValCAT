from abc import abstractmethod

from taattack.transformations.transformation import Transformation


class Replacement(Transformation):
    @abstractmethod
    def _get_replacement_words(self, *args, **kwargs):
        raise NotImplemented()

    def _get_transformed_workload_list(self, workload, idx, is_orig_idx=True):
        curr_idx = workload.orig_idx_2_curr_idx[idx].item() if is_orig_idx else idx
        if curr_idx == -1:
            return []

        replacement_words = self._get_replacement_words(workload, curr_idx)

        transformed_workload_list = [workload.replace(idx, w, is_orig_idx=is_orig_idx) for w in replacement_words]

        return transformed_workload_list
