import json

from steps.process.process import Process


class GetLangCodesProcess(Process):
    def _get_output_desc(self):
        desc = 'output = lang_codes\n' \
               'lang_codes = { sil_code }\n' \
               'sil_code = ? all possible sil codes ?'
        return desc

    def init_for_do(self):
        self.lang_codes_file = self.get('lang_codes')

    def init_for_skip(self):
        raise NotImplementedError

    def _do(self):
        with open(self.lang_codes_file) as f:
            output = json.load(f)
        return output

    def _skip(self):
        raise NotImplementedError