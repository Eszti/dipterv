import sys
import pickle
import networkx as nx
import os
import logging
import utils

class Graphbuilder():
    def __init__(self, cfg):
        section = 'general'
        #Loggin
        self.loglevel = cfg.get(section, 'loglevel')
        loglevel = logging.DEBUG
        if self.loglevel == 'info':
            loglevel = logging.INFO
        logging.basicConfig(stream=sys.stdout, level=loglevel,
                            format='%(asctime)s %(levelname)s '
                                   '%(message)s', datefmt='%Y-%m-%d,%H:%M:%S')
        # that is English for the time being
        from_lang = cfg.get(section, 'from_lang')
        logging.info('Chosen language from which to traslate: {}'.format(from_lang))
        # get to languages
        to_langs_fn = cfg.get(section, 'to_langs_json')
        logging.info('Json file containing language codes to traslate to: {}'.format(to_langs_fn))
        to_langs = utils.get_lang_codes_from_json(to_langs_fn)
        logging.info('Number of languages found in json: {}'.format(len(to_langs)))
        lang_full_mapping = utils.get_language_mappings(cfg.get(section, 'lang_mapping'))
        logging.debug('Number of languages found in language mapping: {}'.format(len(lang_full_mapping)))
        to_langs_mapping =  {k:v for k,v in lang_full_mapping.iteritems() if k in to_langs}
        logging.debug('Number of languages found after mapping: {}'.format(len(to_langs_mapping)))
        # English word list
        self.wordlist = set(utils.get_wordlist(cfg.get(section, 'word_list')))
        logging.info('Number of words found in wordlist: {}'.format(len(self.wordlist)))
        # Creating dictionaries
        dicts = cfg.get(section, "dicts")
        dict_folders = [dict_folder for dict_folder in utils.text_to_list(dicts)]
        logging.info('Dictionaries found: {}'.format(' '.join((dict_folders))))
        self.dictionary_container = utils.create_dictionary_container(from_lang, to_langs_mapping, dict_folders)
        # Graph staff
        section = 'graphs'
        self.printgraph = cfg.getboolean(section, 'print')
        self.png_folder = cfg.get(section, 'png_folder')
        self.gml_folder = cfg.get(section, 'gml_folder')
        self.pickle_folder = cfg.get(section, 'pickle_folder')

    def build_graphs(self):
        rev_wordlist = set([word + '_rev' for word in self.wordlist])
        # if self.loglevel == "debug":
        #     self.dictionary_container.print_content()
        A = nx.DiGraph()
        for dict_source in self.dictionary_container.dict_sources:
            for word in self.wordlist:
                for dictionary in dict_source.dictionaries:
                    trans = dictionary.get(word)
                    if trans is None:
                        continue
                    for t in trans:
                        A.add_edge(word, t)                     # add translation edges
                        rev_trans = dictionary.get_rev(t)
                        for rt in rev_trans:
                            rt_word = rt
                            if rt not in self.wordlist:
                                rev_wordlist.add(rt)            # store as meaning level words
                            else:
                                rt_word = rt + '_rev'           # mark with '_rev' suffix
                            A.add_edge(t, rt_word)              # add backtranslation edges
        logging.info('A graph is created:\n number of nodes: {0}\n number of edges: {1}'
                     .format(A.number_of_nodes(), A.number_of_edges()))
        B = nx.DiGraph()
        for word in self.wordlist:
            for rev_word in rev_wordlist:
                if A.has_node(word) and A.has_node(rev_word):
                    if nx.has_path(A, word, rev_word):
                        paths = [p for p in nx.all_shortest_paths(A, source=word, target=rev_word)]
                        B.add_edge(word, rev_word, weight=len(paths))   # construct bipartite graphs
        logging.info('B graph is created:\n number of nodes: {0}\n number of edges: {1}'
                     .format(B.number_of_nodes(), B.number_of_edges()))
        C = B
        for word in self.wordlist:
            if C.has_node(word):
                C = nx.contracted_nodes(C, word, word + '_rev')
        logging.info('C graph is created:\n number of nodes: {0}\n number of edges: {1}'
                     .format(C.number_of_nodes(), C.number_of_edges()))
        if self.loglevel == 'debug':
            if self.printgraph:
                self.print_graphs(A, B, C)
            self.save_pickle(A, B, C)
        else:
            if self.printgraph:
                self.print_graphs(A=None, B=None, C=C)
            self.save_pickle(A=None, B=None, C=C)
        # if self.loglevel == 'debug':
        #     print 'wordlist: {}'.format(self.wordlist)
        #     print 'rev_wordlist: {}'.format(rev_wordlist)



    def print_graphs(self, A, B, C):
        timestamp = utils.get_timestamp()
        dir_name = os.path.join(self.png_folder, '{0}'.format(timestamp))
        os.makedirs(dir_name)
        logging.info('Printing graphs in png format into: {}'.format(dir_name))
        self._print_graph(A, dir_name, 'A.png')
        self._print_graph(B, dir_name, 'B.png')
        self._print_graph(C, dir_name, 'C.png')

    def save_gml(self, A, B, C):
        timestamp = utils.get_timestamp()
        dir_name = os.path.join(self.gml_folder, '{0}'.format(timestamp))
        os.makedirs(dir_name)
        logging.info('Saving graphs in gml format into: {}'.format(dir_name))
        self._save_gml(A, dir_name, 'A.gml.gz')
        self._save_gml(B, dir_name, 'B.gml.gz')
        self._save_gml(C, dir_name, 'C.gml.gz')

    def save_pickle(self, A, B, C):
        timestamp = utils.get_timestamp()
        dir_name = os.path.join(self.pickle_folder, '{0}'.format(timestamp))
        os.makedirs(dir_name)
        logging.info('Saving graphs in binary format into: {}'.format(dir_name))
        self._save_pickle(A, dir_name, 'A.pickle')
        self._save_pickle(B, dir_name, 'B.pickle')
        self._save_pickle(C, dir_name, 'C.pickle')

    def _save_pickle(self, graph, dir_name, file_name):
        if graph is not None:
            file_path = os.path.join(dir_name, file_name)
            with open(file_path, 'w') as f:
                pickle.dump(graph, f)

    def _save_gml(self, graph, dir_name, file_name):
        if graph is not None:
            file_path = os.path.join(dir_name, file_name)
            nx.write_gml(graph, file_path)

    def _print_graph(self, graph, dir_name, file_name):
        if graph is not None:
            file_path = os.path.join(dir_name, file_name)
            pydot_graph = utils.graph_to_pydot(graph)
            pydot_graph.write_png(file_path)

    def dummy(self):
        print self.wordlist
        self.dictionary_container.print_content()


def main():
    cfg_file = sys.argv[1] if len(sys.argv) > 1 else None
    cfg = utils.get_cfg(cfg_file)
    gb = Graphbuilder(cfg)
    # gb.dummy()
    gb.build_graphs()

if __name__ == '__main__':
    main()