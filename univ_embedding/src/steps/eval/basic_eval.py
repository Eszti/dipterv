import numpy as np
from sklearn.preprocessing import normalize

from steps.eval.langwise_eval import LangWiseEvaluation


class BasicEvaluation(LangWiseEvaluation):
# input : ( [lang : swad_list, emb_full (norm), emb_fn, not_found_list, T], univ(norm))

    def _init_for_eval(self):
        pass

    def get_header(self):
        headers = ['lang_code',
                   'diff_full(orig, trans)',
                   'diff_full(orig, univ)',
                   'diff_full(trans, univ)',
                   'cos_full(orig, trans)',
                   'cos_full(orig, univ)',
                   'cos_full(trans, univ)',

                   'diff_orig(orig, trans)',
                   'diff_orig(orig, univ)',
                   'diff_orig(trans, univ)',
                   'cos_orig(orig, trans)',
                   'cos_orig(orig, univ)',
                   'cos_orig(trans, univ)',

                   'diff_new(orig, trans)',
                   'diff_new(orig, univ)',
                   'diff_new(trans, univ)',
                   'cos_new(orig, trans)',
                   'cos_new(orig, univ)',
                   'cos_new(trans, univ)'
                   ]
        return headers

    def calc_values(self, orig, trans, univ, orig_cos, trans_cos, univ_cos):
        ret = []
        # Diff(orig, trans): frob norm
        ret.append(np.linalg.norm(orig - trans))
        # Diff(orig, univ): frob norm
        ret.append(np.linalg.norm(orig - univ))
        # Diff(trans, univ): frob norm
        ret.append(np.linalg.norm(trans - univ))
        # Correlation between orig and translated cos sim mx-s
        ret.append(self._corr_cos_sims(orig_cos, trans_cos))
        # Correlation between orig and univ cos sim mx-s
        ret.append(self._corr_cos_sims(orig_cos, univ_cos))
        # Correlation between trans and univ cos sim mx-s
        ret.append(self._corr_cos_sims(univ_cos, trans_cos))
        return ret

    def get_row_values(self, lang, list, univ, univ_cos):
        row = []
        orig_full = list[1]
        orig_full_cos = self._get_cos_sim_mx(orig_full)
        T = list[4]
        translation = np.dot(orig_full, T)
        trans_full = normalize(translation)
        trans_full_cos = self._get_cos_sim_mx(trans_full)
        row += self.calc_values(orig_full, trans_full, univ, orig_full_cos, trans_full_cos, univ_cos)

        not_found_list = list[3]
        orig_orig = np.delete(np.delete(orig_full, not_found_list, axis=0), not_found_list, axis=1)
        orig_orig_cos = np.delete(np.delete(orig_full_cos, not_found_list, axis=0), not_found_list, axis=1)
        trans_orig = np.delete(np.delete(trans_full, not_found_list, axis=0), not_found_list, axis=1)
        trans_orig_cos = np.delete(np.delete(trans_full_cos, not_found_list, axis=0), not_found_list, axis=1)
        univ_orig = np.delete(np.delete(univ, not_found_list, axis=0), not_found_list, axis=1)
        univ_orig_cos = np.delete(np.delete(univ_cos, not_found_list, axis=0), not_found_list, axis=1)
        row += self.calc_values(orig_orig, trans_orig, univ_orig, orig_orig_cos, trans_orig_cos, univ_orig_cos)

        orig_new = orig_full[not_found_list, :][:, not_found_list]
        orig_new_cos = orig_full_cos[not_found_list, :][:, not_found_list]
        trans_new = trans_full[not_found_list, :][:, not_found_list]
        trans_new_cos = trans_full_cos[not_found_list, :][:, not_found_list]
        univ_new = univ[not_found_list, :][:, not_found_list]
        univ_new_cos = univ_cos[not_found_list, :][:, not_found_list]
        row += self.calc_values(orig_new, trans_new, univ_new, orig_new_cos, trans_new_cos, univ_new_cos)
        return row