from steps.step import Step


class Process(Step):
    def __init__(self, name, config, starttime):
        super(Process, self).__init__(name, config, starttime)

    def run(self, input, do=False):
        return self._run(input, do)

    def _run(self, input, do=False):
        self.input = input
        if do:
            return self.do()
        else:
            return self.skip()

    def init_for_do(self):
        raise NotImplementedError

    def init_for_skip(self):
        raise NotImplementedError

    def _do(self):
        raise NotImplementedError

    def _skip(self):
        raise NotImplementedError

    def do(self):
        self.init_for_do()
        return self._do()

    def skip(self):
        self.init_for_skip()
        return self._skip()