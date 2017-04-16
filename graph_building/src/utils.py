from ConfigParser import ConfigParser
import os
import json
import time
import dictionary as dic
import pydot

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

def get_lang_codes_from_json(fn):
    with open(fn) as f:
        to_langs = json.load(f)
    return to_langs

def get_language_mappings(fn):
    mapping = dict()
    with open(fn) as f:
        lines = f.read().splitlines()
        for line in lines:
            fileds = line.split('\t')
            mapping[fileds[0]] = fileds[1]
    return mapping

def text_to_list(text, delim='|'):
    return text.split(delim)

def get_wordlist(filename):
    with open(filename) as f:
        lines = f.readlines()
        words = [line.strip() for line in lines]
    return words

def process_tsv_dict_file(filename, dictionary=dict(), rev_dictionary=dict()):
    with open(filename) as f:
        lines = f.readlines()
        for line in lines:
            fields = line.strip().decode('utf-8').split('\t')
            w = fields[0]
            trans = fields[1]
            _put_to_dict(w, trans, dictionary)
            _put_to_dict(trans, w, rev_dictionary)
    return dictionary, rev_dictionary

def _put_to_dict(w, trans, dictionary):
    if w not in dictionary.keys():
        dictionary[w] = set()
    dictionary[w].add(trans)

def create_dictionary_container(from_lang, to_langs, dict_folders):
    return dic.DictionaryContainer(from_lang, to_langs, dict_folders)

def graph_to_pydot(G):
    graph = pydot.Dot(graph_type='digraph')
    for e, f, data in G.edges(data=True):
        if 'weight' in data.keys():
            graph.add_edge(pydot.Edge(e, f, label=data['weight']))
        else:
            graph.add_edge(pydot.Edge(e, f))
    return graph

def get_timestamp():
    time_str = time.strftime("%H%M")
    date_str = time.strftime("%Y%m%d")
    return '{0}_{1}'.format(date_str, time_str)
