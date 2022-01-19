from enum import Enum


class Status(Enum):
    SKIPPED = 0
    SEARCHING = 1
    SUCCESSFUL = 2
    FAILED = 3

    def __str__(self):
        return self._name_
