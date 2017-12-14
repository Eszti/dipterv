import logging
import time
import os
import json
import numpy as np

loglevel = logging.INFO

def get_timestamp_str():
    time_str = time.strftime("%H%M_%S")
    date_str = time.strftime("%Y%m%d")
    return '{0}_{1}'.format(date_str, time_str)

output_dir = os.path.join('output', '{}'.format(get_timestamp_str()))
os.makedirs(output_dir)

logging.basicConfig(filename=os.path.join(output_dir, 'log.txt'),
                    level=loglevel,
                    format='%(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p')

def print_debug(str):
    logging.debug(str)

def print_verbose(str):
    logging.info(str)

def save(T, lc, precs, i=None):
    def _save(data, name, id, as_json=True):
        fn = os.path.join(output_dir, '{0}_{1}.json'.format(name, id))
        with open(fn, 'w') as f:
            if as_json:
                json.dump(data, f)
            else:
                np.savez(f, T=data)
        print_verbose('saved: {0}'.format(fn))
    if i is None:
        id = 'final'
    else:
        id = str(i)
    _save(lc, 'lc', id)
    _save(precs, 'precs', id)
    _save(T,  'T', id)
