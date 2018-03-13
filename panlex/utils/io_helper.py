import csv
import errno
import pickle

import sys

import re
from shutil import copyfile

import os
import json

import strings

sys.path.insert(0, 'utils')

def log_or_print(s, logger=False):
    if not logger:
        print(s)
    else:
        logger.info(s)

def checkdir(filename):
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise OSError

def save_pickle(data, filename):
    checkdir(filename)
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def load_pickle(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

def save_json(data, filename):
    checkdir(filename)
    with open(filename, 'wt') as f:
        json.dump(data, f)

def list_to_csv(data, filename, delim='\t'):
    checkdir(filename)
    with open(filename, 'wt') as f:
        wr = csv.writer(f, dialect='excel', delimiter=delim)
        wr.writerows(data)

def copy_files(output_dir, orig_files, logger=None):
    for i, orig_fn in enumerate(orig_files):
        basename = os.path.basename(orig_fn)
        dest_fn = os.path.join(output_dir, basename)
        if i == 0:
            checkdir(dest_fn)
        copyfile(orig_fn, dest_fn)
        if logger is not None:
            logger.debug('{} is copied to {}'.format(orig_fn, dest_fn))
        else:
            logger.debug('{} is copied to {}'.format(orig_fn, dest_fn))

def tsv_into_arrays(fn, delim):
    with open(fn) as csvfile:
        spamreader = csv.reader(csvfile, delimiter=delim)
        init = False
        col_nb = None
        data_array = None
        for row in spamreader:
            if not init:
                col_nb = len(row)
                data_array = [[] for _ in range(col_nb)]
                init = True
            for i in range(col_nb):
                data_array[i].append((float(row[i])))
    return data_array

def get_matching_files(folder, pattern):
    list_of_files = os.listdir(folder)
    found_files = []
    for file in list_of_files:
        s = re.search(pattern, file)
        if s is not None:
            found_files.append(file)
    return found_files
