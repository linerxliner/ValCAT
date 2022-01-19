import numpy as np
from tqdm import tqdm

from taattack.goals.status import Status
from taattack.search_methods.search_method import SearchMethod
from taattack.transformations import EncoderDecoderInsertion, EncoderDecoderReplacement


class GreedySpanImportanceRanking(SearchMethod):
    def __init__(self, trans, goal, constraints=None, num_order=3):
        self._num_order = num_order

        super().__init__(trans, goal, constraints=constraints)

    def _get_idx_ranking(self, workload, num_order=3):
        num_words = len(workload.words)
        num_order = min(num_order, num_words)
        idx_randking = []

        for order in range(1, num_order + 1):
            workload_list = [workload.replace(i, self._goal.victim.unk_token, num_replaced=order) for i in range(num_words - order + 1)]
            word_scores = self._goal.get_scores(workload_list).tolist()
            idx_randking.extend(zip(list(range(num_words - order + 1)), [order] * (num_words - order + 1), word_scores))

        idx_randking = sorted(idx_randking, key=lambda t: t[2], reverse=True)
        idx_randking = list(map(lambda t: (t[0], t[1]), idx_randking))

        return idx_randking

    def search(self, workload):
        best_result = self._goal([workload])[0]
        if best_result.status == Status.SUCCESSFUL:
            best_result.status = Status.SKIPPED
            return best_result

        idx_randking = self._get_idx_ranking(workload, num_order=self._num_order)
        idx_randking = self._constraints.pre_filter(idx_randking, workload)

        pbar = tqdm(idx_randking)
        for idx, order in pbar:
            workload_list = []

            for t in self._trans.trans:
                if isinstance(t, EncoderDecoderInsertion):
                    workload_list.extend(t(workload, idx=idx))
                elif isinstance(t, EncoderDecoderReplacement) and t.replace_window == order:
                    workload_list.extend(t(workload, idx=idx + order - 1))
            if len(workload_list) == 0:
                continue

            filtered_workload_list = self._constraints.text_filter(workload_list)
            if len(filtered_workload_list) == 0:
                best_result.failed_reason = workload_list[0].failed_reason
                best_result.status = Status.FAILED
                break
            workload_list = filtered_workload_list

            results = self._goal(workload_list)
            statuses = [r.status for r in results]
            if Status.SUCCESSFUL in statuses:
                results = list(filter(lambda r: r.status == Status.SUCCESSFUL, results))
                best_idx = np.array([r.workload.sim for r in results]).argmax()
                best_result = results[best_idx]
            else:
                best_idx = np.array([r.score for r in results]).argmax()
                if results[best_idx].score > best_result.score:
                    best_result = results[best_idx]
            pbar.set_description(f'Score: {best_result.score}')
            if best_result.status == Status.SUCCESSFUL or Status.FAILED in statuses:
                break

            workload = best_result.workload

        if best_result.status == Status.SEARCHING:
            best_result.failed_reason = 'Tried all perturbations'
            best_result.status = Status.FAILED

        return best_result
