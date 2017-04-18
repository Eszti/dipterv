from steps.step import Step


class Filter(Step):
    def __init__(self, name, config, starttime):
        super(Filter, self).__init__(name, config, starttime)

    def run(self, input, do=False):
        if do:
            return self.filter(input)
        else:
            return input

    def filter(self, input):
        raise NotImplementedError