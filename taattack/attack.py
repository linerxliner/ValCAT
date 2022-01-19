import time


class Attack:
    def __init__(self, search_method):
        self._search_method = search_method

    def attack(self, workload):
        start = time.process_time()
        result = self._search_method.search(workload)
        result.time = time.process_time() - start

        return result
