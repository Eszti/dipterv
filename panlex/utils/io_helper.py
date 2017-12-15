import errno
import pickle
from shutil import copyfile

import os


def _checkdir(filename):
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise OSError

def save_pickle(data, filename):
    _checkdir(filename)
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def copy_files(output_dir, orig_files, logger=None):
    for i, orig_fn in enumerate(orig_files):
        basename = os.path.basename(orig_fn)
        dest_fn = os.path.join(output_dir, basename)
        if i == 0:
            _checkdir(dest_fn)
        copyfile(orig_fn, dest_fn)
        if logger is not None:
            logger.debug('{} is copied to {}'.format(orig_fn, dest_fn))
        else:
            logger.debug('{} is copied to {}'.format(orig_fn, dest_fn))
