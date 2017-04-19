import logging

from steps.step import Step


class Process(Step):
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
        logging.info('Do function is called')
        self.init_for_do()
        return self._do()

    def skip(self):
        logging.info('Skip function is called')
        self.init_for_skip()
        return self._skip()