from steps.step import Step


class Filter(Step):
    def __init__(self, name, genparams):
        super(Filter, self).__init__(name, genparams)

    def init_filter(self):
        pass

    def _run(self, input, do=False):
        self.init_filter()
        if do:
            return self.filter(input)
        else:
            return input

    def filter(self, input):
        raise NotImplementedError