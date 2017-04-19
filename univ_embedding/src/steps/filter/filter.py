from steps.step import Step


class Filter(Step):
    def __init__(self, name, genparams):
        super(Filter, self).__init__(name, genparams)

    def _run(self, input, do=False):
        if do:
            return self.filter(input)
        else:
            return input

    def filter(self, input):
        raise NotImplementedError