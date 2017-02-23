import utils
import os
import sys
import time
import networkx as nx

class Graphbuilder():
    def __init__(self, cfg):
        section = 'general'
        from_lang = cfg.get(section, 'from_lang')
        to_langs = utils.text_to_list(cfg.get(section, 'to_langs'))
        self.wordlist = set(utils.get_wordlist(cfg.get(section, 'word_list')))
        dicts = cfg.get(section, "dicts")
        dict_folders = [os.path.join('res', 'dicts', dict_folder) for dict_folder in utils.text_to_list(dicts)]
        self.dictionary_container = utils.create_dictionary_container(from_lang, to_langs, dict_folders)
        self.loglevel = cfg.get(section, 'loglevel')

    def build_graphs(self):
        rev_wordlist = set([word + '_rev' for word in self.wordlist])
        A = nx.DiGraph()
        for dict_source in self.dictionary_container.dict_sources:
            for word in self.wordlist:
                for dictionary in dict_source.dictionaries:
                    trans = dictionary.get(word)
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
        B = nx.DiGraph()
        for word in self.wordlist:
            for rev_word in rev_wordlist:
                if nx.has_path(A, word, rev_word):
                    paths = [p for p in nx.all_shortest_paths(A, source=word, target=rev_word)]
                    B.add_edge(word, rev_word, weight=len(paths))   # construct bipartite graphs
        C = B
        for word in self.wordlist:
            C = nx.contracted_nodes(C, word, word + '_rev')
        print B.edges(data=True)
        print C.edges(data=True)
        if self.loglevel == 'debug':
            self.print_graphs(A, B, C)
        else:
            self.print_graphs(A=None, B=None, C=C)
        print rev_wordlist

    def print_graphs(self, A, B, C):
        time_str = time.strftime("%H%M")
        date_str = time.strftime("%Y%m%d")
        dir_name = os.path.join('graphs', '{0}_{1}'.format(date_str, time_str))
        os.makedirs(dir_name)
        self._print_graph(A, dir_name, 'A.png')
        self._print_graph(B, dir_name, 'B.png')
        self._print_graph(C, dir_name, 'C.png')

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