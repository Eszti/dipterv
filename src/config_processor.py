from ConfigParser import ConfigParser
import os


def get_cfg(cfg_file):
    cfg_files = [os.path.join(os.environ['FOURLANGPATH'], 'conf/default.cfg')]
    not_found = [fn for fn in cfg_files if not os.path.exists(fn)]
    if cfg_file is not None:
        cfg_files.append(cfg_file)
    if not_found:
        raise Exception("cfg file(s) not found: {0}".format(not_found))
    cfg = ConfigParser(os.environ)
    cfg.read(cfg_files)
    return cfg
