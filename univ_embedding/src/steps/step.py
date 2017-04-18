__metaclass__ = type
class Step():
    def __init__(self, name, config, starttime):
        self.name = name
        self.config = config
        self.starttime = starttime

    def run(self, input, do=False):
        raise NotImplementedError