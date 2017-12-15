from base.loggable import Loggable


class TrainMModel(Loggable):
    def __init__(self, data_config, train_config):
        Loggable.__init__(self)
        self.train_config = train_config


    def train(self):
        pass

    def test(self):
        pass

    def run(self):
        pass