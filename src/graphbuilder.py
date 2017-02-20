from ConfigParser import ConfigParser
import sys

class Graphbuilder():
    def __init__(self, cfg):


def main():
    cfg_file = sys.argv[1] if len(sys.argv) > 1 else None
    cfg = get_cfg(cfg_file)
