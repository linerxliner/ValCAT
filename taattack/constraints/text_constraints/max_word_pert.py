from .text_constraint import TextConstraint


class MaxWordPert(TextConstraint):
    def __init__(self, max_num=None, max_pct=None):
        self._max_num = max_num
        self._max_pct = max_pct

    def _check(self, workload):
        num_word_pert = workload.num_word_pert

        if self._max_num:
            r = num_word_pert <= self._max_num
            if not r:
                workload.failed_reason = f'Num of word perturbation ({num_word_pert}) exceeds {self._max_num}'
        elif self._max_pct:
            num_words = len(workload.orig.words)
            pct_word_pert = num_word_pert / num_words
            r = pct_word_pert <= self._max_pct
            if not r:
                workload.failed_reason = f'Pct of word perturbation ({pct_word_pert:.2f}) exceeds {self._max_pct}'
        else:
            r = True

        return r
