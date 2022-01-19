from .status import Status

class Result:
    def __init__(self, workload, status, score=None):
        self.workload = workload
        self.status = status
        self.score = score

        self.time = None
        self.failed_reason = workload.failed_reason

    def __str__(self):
        status_sec = f'[{self.status}:{self.workload.label}->{self.workload.probs.argmax()}]'
        if self.status == Status.FAILED:
            status_sec = status_sec[:-1] + f':{self.failed_reason}]'
        time_sec = f'({self.time:.2f}s)'
        orig_text = self.workload.orig.full_text
        adv_text = self.workload.full_text

        return f'{status_sec}{time_sec}\n{orig_text}\n{adv_text}'

    def to_dict(self):
        d = {
            'orig_workload': self.workload.orig.to_dict(),
            'adv_workload': self.workload.to_dict(),
            'status': str(self.status),
            'score': self.score,
        }
        if self.time is not None:
            d['time'] = self.time
        if self.failed_reason is not None:
            d['failed_reason'] = self.failed_reason

        return d