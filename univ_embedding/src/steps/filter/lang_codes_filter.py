from steps.filter.filter import Filter

# input : list of sil codes
# output : list of sil codes

class LangCodesFilter(Filter):
    def _get_output_desc(self):
        return 'input: list of sil codes\n' \
               'output: list of sil codes'

    def filter(self, input):
        return input