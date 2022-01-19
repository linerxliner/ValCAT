class TransformationBundle:
    def __init__(self, trans):
        self.trans = trans

    def __call__(self, *args, **kwargs):
        transformed_workload_list = []

        for t in self.trans:
            transformed_workload_list.extend(t(*args, **kwargs))

        return transformed_workload_list
