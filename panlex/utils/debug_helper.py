import numpy as np

from io_helper import log_or_print


def get_smalls(T, limit, nb, logger=False):
    U, s, V = np.linalg.svd(T, full_matrices=True, compute_uv=True)
    small = np.where(s < limit)
    log_or_print('T_{0}\t<{1} : {2}'.format(nb, limit, len(small[0])), logger)