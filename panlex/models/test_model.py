import sys

sys.path.insert(0, 'utils')

from plot_helper import plot_progress
from base.loggable import Loggable


class TestModel(Loggable):
    def __init__(self, input_dir):
        Loggable.__init__(self)
        self.input_dir = input_dir

    def plot_progress(self):
        plot_progress(logger=self.logger, input_folder=self.input_dir)
