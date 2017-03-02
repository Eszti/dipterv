import os
import utils
import logging

class Dictionary:
    def __init__(self, from_lang, to_lang, dict_folder):
        self.id = (from_lang, to_lang)
        self.from_lang = from_lang
        self.to_lang = to_lang

        try:
            dict_name = from_lang + '_' + to_lang + '.txt'
            filename = os.path.join(dict_folder, dict_name)
            self.dict = utils.process_tsv_dict_file(filename)

            rev_dict_name = to_lang + '_' + from_lang + '.txt'
            filename = os.path.join(dict_folder, rev_dict_name)
            self.rev_dict = utils.process_tsv_dict_file(filename)

        except:
            logging.error("could not process dictionary file: {0}".format(filename))

    def get(self, word):
        try:
            return self.dict[word]
        except:
            logging.warning('OOV found: {}'.format(word))
            return None

    def get_rev(self, word):
        try:
            return self.rev_dict[word]
        except:
            logging.warning('OOV found is rev_dict: {}'.format(word))
            return None

    def print_content(self):
        print "\t{0}_{1}:".format(self.from_lang, self.to_lang)
        print self.dict
        print "\t{0}_{1}:".format(self.to_lang, self.from_lang)
        print self.rev_dict


class DictionarySource:
    def __init__(self, from_lang, to_langs, dict_folder):
        _, self.name = os.path.split(dict_folder)
        self.dictionaries = []

        for to in to_langs:
            logging.debug('processing dictionary: {0}-{1} in folder: {2}'.format(from_lang, to, dict_folder))
            new_dict = Dictionary(from_lang, to, dict_folder)
            self.dictionaries.append(new_dict)

    def get(self, word):
        trans = []
        for dictionary in self.dictionaries:
            trans.append(dictionary.get(word))
        return trans

    def get_rev(self, word):
        trans_rev = []
        for dictionary in self.dictionaries:
            trans_rev.append(dictionary.get_rev(word))
        return trans_rev

    def print_content(self):
        print "{0}".format(self.name)
        for dictionary in self.dictionaries:
            dictionary.print_content()

class DictionaryContainer:
    def __init__(self, from_lang, to_langs, dict_folders):
        self.dict_sources = []
        for dict_folder in dict_folders:
            logging.info('loading dictionary source: {0}'.format(dict_folder))
            new_ds = DictionarySource(from_lang, to_langs, dict_folder)
            self.dict_sources.append(new_ds)

    def get(self, word):
        trans = []
        for dict_source in self.dict_sources:
            trans.append(dict_source.get(word))
        return trans

    def get_rev(self, word):
        trans_rev = []
        for dict_source in self.dict_sources:
            trans_rev.append(dict_source.get_rev(word))
        return trans_rev

    def print_content(self):
        for dict_s in self.dict_sources:
            dict_s.print_content()
