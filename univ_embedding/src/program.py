import argparse
import logging
import time
from ConfigParser import ConfigParser

import os

from data_structures import GeneralParams
from find_univ_proc import FindUnivProcess
from helpers import create_timestamped_dir
from steps.filter.embed_filter import EmbedFilter
from steps.filter.lang_codes_filter import LangCodesFilter
from steps.filter.swad_filter import SwadFilter
from steps.process.get_embed_proc import GetEmbedProcess
from steps.process.get_lang_codes_proc import GetLangCodesProcess
from steps.process.get_swad_proc import GetSwadProcess
from translate_emb_proc import TranslateEmbProcess

# 1.:   get_lang_codes_proc     GetLangCodesProcess     Process
#       long_codes_filter       LangCodesFilter         Filter
# 2 :   get_swad_proc           GetSwadProcess          Process
#       swad_filter             SwadFilter              Filter
# 3 :   get_embed_proc          GetEmbedProcess         Process
#       embed_filter            SwadFilter              Filter
# 4 :   translate_emb_proc      TranslateEmbProcess     Process
#       translated_filter       TranslatedFilter        Filter
# 5 :   find_univ_proc          FindUnivProcess         Process
#       univ_filter             UnivFilter              Filter
# 6 :   evaluation_proc         EvaluationProcess       Process


def main(config_file, start, finish, output_dir):
    cfg = ConfigParser(os.environ)
    cfg.read(config_file)
    genparams = GeneralParams(starttime, cfg, output_dir)
    steps = []
    do_flag = False
    if start == None:
        start = 1
    if start == 1:
        do_flag = True
    steps.append((GetLangCodesProcess('get_lang_codes_proc', genparams), do_flag))     # 1
    steps.append((LangCodesFilter('lang_codes_filter', genparams), do_flag))
    if start == 2:
        do_flag = True
    steps.append((GetSwadProcess('get_swad_proc', genparams), do_flag))                # 2
    steps.append((SwadFilter('swad_filter', genparams), do_flag))
    if start == 3:
        do_flag = True
    steps.append((GetEmbedProcess('get_embed_proc', genparams), do_flag))              # 3
    steps.append((EmbedFilter('embed_filter', genparams), do_flag))
    if start == 4:
        do_flag = True
    steps.append((TranslateEmbProcess('translate_emb_proc', genparams), do_flag))      # 4
    if start == 5:
        do_flag = True
    steps.append((FindUnivProcess('find_univ_proc', genparams), do_flag))              # 5
    input = None
    i = 1
    for (step, do) in steps:
        output = step.run(input, do)
        input = output
        if finish == i:
            logging.info('Finishing now, finish was set to {}'.format(finish))
            break
        i += 1
    output = input

if __name__ == '__main__':
    starttime = int(round(time.time()))
    os.nice(20)
    parser = argparse.ArgumentParser(description='Pipeline for creating universal embedding.')
    parser.add_argument('-c', '--config', type=str, dest='config_file', help='config file')
    parser.add_argument('-s', '--start', type=int, dest='start',
                        help='start processing from this step, steps before will be loaded by the skip method')
    parser.add_argument('-f', '--finish', type=int, dest='finish',
                        help='finish processing after this step, steps after will be not be executed')
    args = parser.parse_args()

    output_dir = create_timestamped_dir('output')

    logfile = os.path.join(output_dir, 'log.txt')
    logging.basicConfig(filename=logfile, level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d,%H:%M:%S')
    logging.info('Starttime: {}'.format(starttime))
    logging.info('Start: {0} - Finish: {1}'.format(args.start, args.finish))
    main(args.config_file,  args.start, args.finish, output_dir)