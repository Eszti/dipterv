from steps.filter.filter import Filter


class SwadFilter(Filter):
    def filter(self, input):
        output = dict()
        for lang, swad_list in input.iteritems():
            filtered_list = []
            if swad_list is not None:
                for entry_list in swad_list:
                    if entry_list is not None:
                        found = False
                        for entry in entry_list:
                            if ' ' not in entry:
                                word = entry.lower()
                                found = True
                                break
                        if not found:
                            word = None
                    else:
                        word = None
                    filtered_list.append(word)
                output[lang] = filtered_list
        return output