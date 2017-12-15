import logging

class Loggable():
    def __init__(self):
        self.logger = logging.getLogger(str(self.__class__))