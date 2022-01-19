from abc import ABC


class Goal(ABC):
    def __init__(self, victim):
        self.victim = victim
