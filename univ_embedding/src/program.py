import argparse
import time
import logging
import os
from ConfigParser import ConfigParser
from steps.filter.swad_filter import SwadFilter
from steps.process.get_lang_codes import GetLangCodesProcess
from steps.process.get_swad_proc import GetSwadProcess


def main(config_file, starttime):
    cfg = ConfigParser(os.environ)
    cfg.read(config_file)
    steps = []
    steps.append((GetLangCodesProcess('get_lang_codes', cfg, starttime), True))
    steps.append((GetSwadProcess('get_swad_proc', cfg, starttime), True))
    steps.append((SwadFilter('swad_filter', cfg, starttime), True))
    input = None
    for (step, do) in steps:
        output = step.run(input, do)
        input = output
    output = input

if __name__ == '__main__':
    starttime = int(round(time.time()))
    os.nice(20)
    parser = argparse.ArgumentParser(description='Pipeline for creating universal embedding.')
    parser.add_argument('-c', '--config', type=str, dest='config_file', help='config file')
    args = parser.parse_args()

    time_str = time.strftime("%H%M_%S")
    date_str = time.strftime("%Y%m%d")
    output_dir = os.path.join('output', '{0}_{1}'.format(date_str, time_str))
    os.makedirs(output_dir)

    logfile = os.path.join(output_dir, 'log.txt')
    logging.basicConfig(filename=logfile, level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d,%H:%M:%S')

    main(args.config_file, starttime)