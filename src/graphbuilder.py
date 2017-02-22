import utils
import os
import sys

class Graphbuilder():
    def __init__(self, cfg):
        section = 'general'
        from_lang = cfg.get(section, 'from_lang')
        to_langs = utils.text_to_list(cfg.get(section, 'to_langs'))
        self.wordlist = utils.get_wordlist(cfg.get(section, 'word_list'))
        dicts = cfg.get(section, "dicts")
        dict_folders = [os.path.join('res', 'dicts', dict_folder) for dict_folder in utils.text_to_list(dicts)]
        self.dictionary_container = utils.create_dictionary_container(from_lang, to_langs, dict_folders)

    def dummy(self):
        print self.wordlist
        self.dictionary_container.print_content()


def main():
    cfg_file = sys.argv[1] if len(sys.argv) > 1 else None
    cfg = utils.get_cfg(cfg_file)
    gb = Graphbuilder(cfg)
    gb.dummy()

if __name__ == '__main__':
    main()