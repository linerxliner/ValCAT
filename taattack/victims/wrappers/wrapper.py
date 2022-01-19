import numpy as np
from abc import ABC, abstractmethod

from taattack.workload import Workload


class Wrapper(ABC):
    sep_token = ''

    def __call__(self, workload_list, batch_size=64):
        logits = []

        if isinstance(workload_list[0], Workload):
            text_list = []

            text_list.append([w.text for w in workload_list])
            if workload_list[0].extra_text:
                text_list.append([w.extra_text for w in workload_list])
            if len(text_list) == 2 and (not workload_list[0].text_before_extra):
                text_list[0], text_list[1] = text_list[1], text_list[0]
        else:
            text_list = [workload_list]

        bz = batch_size
        while bz > 0:
            try:
                logits.clear()
                for i in range((len(text_list[0]) + bz - 1) // bz):
                    batch = [texts[i * bz:(i + 1) * bz] for texts in text_list]
                    logits.append(self._forward(batch))
                break
            except RuntimeError as e:
                print(e)
            bz /= 2

        return np.concatenate(logits, axis=0)

    @abstractmethod
    def _forward(self, text_list):
        raise NotImplementedError
