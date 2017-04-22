import json

from steps.process.process import Process

# input : file containing silcodes
# output : list of sil codes

class GetLangCodesProcess(Process):
    def _get_output_desc(self):
        return 'input : file containing silcodes\n' \
               'output : list of sil codes'

    def init_for_do(self):
        self.lang_codes_file = self.get('lang_codes')

    def _do(self):
        with open(self.lang_codes_file) as f:
            output = json.load(f)
        return output