import numpy as np
import scipy.stats
from sklearn.preprocessing import normalize

from eval import Eval
from helpers import filter_list, retain_list

# input : ( [lang : swad_list, emb_full (norm), not_found_list, T], univ(norm))

class BasicEval(Eval):
    def calc_values(self, orig, trans, univ, univ_cos):
        ret = []
        # Diff(orig, trans): frob norm
        ret.append(np.linalg.norm(orig - trans))
        # Diff(orig, univ): frob norm
        ret.append(np.linalg.norm(orig - univ))
        # Diff(trans, univ): frob norm
        ret.append(np.linalg.norm(trans - univ))
        orig_cos_flat = np.ndarray.flatten(self._get_cos_sim_mx(orig))
        trans_cos_flat = np.ndarray.flatten(self._get_cos_sim_mx(trans))
        univ_cos_flat = np.ndarray.flatten(univ_cos)
        # Correlation between orig and translated cos sim mx-s
        ret.append(scipy.stats.pearsonr(orig_cos_flat, trans_cos_flat))
        # Correlation between orig and univ cos sim mx-s
        ret.append(scipy.stats.pearsonr(orig_cos_flat, univ_cos_flat))
        # Correlation between trans and univ cos sim mx-s
        ret.append(scipy.stats.pearsonr(univ_cos_flat, trans_cos_flat))
        return ret

    def evalute(self, input):
        headers = ['lang_code',
                   'diff(orig(full), trans)',
                   'diff(orig(full), univ)',
                   'diff(trans, univ)',
                   'cos(orig(full), trans)',
                   'cos(orig(full), univ)',
                   'cos(trans, univ)',

                   'diff_orig(orig(full), trans)',
                   'diff_orig(orig(full), univ)',
                   'diff_orig(trans, univ)',
                   'cos_orig(orig, trans)',
                   'cos_orig(orig, univ)',
                   'cos_orig(trans, univ)',

                   'diff_new(orig(full), trans)',
                   'diff_new(orig(full), univ)',
                   'diff_new(trans, univ)',
                   'cos_new(orig, trans)',
                   'cos_new(orig, univ)',
                   'cos_new(trans, univ)'
                   ]
        data = input(0)
        univ = input(1)
        results = []
        results.append(headers)
        row = []
        for lang, list in data.iteritems():
            row.append(lang)
            orig_full = list[1]
            T = list[3]
            translation = np.dot(orig_full, T)
            trans_full = normalize(translation)
            row += self.calc_values(orig_full, trans_full, univ)

            not_found_list = list[2]
            orig_orig = filter_list(orig_full, not_found_list)
            trans_orig = filter_list(trans_full, not_found_list)
            univ_orig = filter_list(univ, not_found_list)
            row += self.calc_values(orig_orig, trans_orig, univ_orig)

            orig_new = retain_list(orig_full, not_found_list)
            trans_new = retain_list(trans_full, not_found_list)
            univ_new = retain_list(univ, not_found_list)
            row += self.calc_values(orig_new, trans_new, univ_new)

            results.append(row)
        return results