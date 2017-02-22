from ConfigParser import ConfigParser
import os
import dictionary as dic

def get_cfg(cfg_file):
    pwd = os.path.dirname(os.path.realpath(__file__))
    cfg_files = [os.path.join(pwd, '../conf/default.cfg')]
    not_found = [fn for fn in cfg_files if not os.path.exists(fn)]
    if cfg_file is not None:
        cfg_files.append(cfg_file)
    if not_found:
        raise Exception("cfg file(s) not found: {0}".format(not_found))
    cfg = ConfigParser(os.environ)
    cfg.read(cfg_files)
    return cfg

def text_to_list(text, delim='|'):
    return text.split(delim)

def get_wordlist(filename):
    words = []
    with open(filename) as f:
        lines = f.readlines()
        words = [line.strip() for line in lines]
    return words

def process_tsv_dict_file(filename):
    dictionary = {}
    with open(filename) as f:
        lines = f.readlines()
        for line in lines:
            fields = line.strip().decode('utf-8').split('\t')
            w = fields[0]
            trans = fields[1:]
            dictionary[w] = trans
    return dictionary

def create_dictionary_container(from_lang, to_langs, dict_folders):
    return dic.DictionaryContainer(from_lang, to_langs, dict_folders)