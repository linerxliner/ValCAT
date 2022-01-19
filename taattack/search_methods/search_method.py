from taattack.constraints import ConstraintBundle
from taattack.transformations import TransformationBundle


class SearchMethod:
    def __init__(self, trans, goal, constraints=None):
        self._trans = TransformationBundle(trans)
        self._goal = goal
        self._constraints = ConstraintBundle(constraints)
