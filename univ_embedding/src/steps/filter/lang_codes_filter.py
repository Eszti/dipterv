from steps.filter.filter import Filter


class LangCodesFilter(Filter):
    def _get_output_desc(self):
        desc = 'output = lang_codes\n' \
               'lang_codes = { sil_code }\n' \
               'sil_code = ? all possible sil codes ?'
        return desc

    def filter(self, input):
        return input