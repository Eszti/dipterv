import time

import numpy as np
import os

def find_all_indices(list, val):
    idxs = [i for i, elem in enumerate(list) if elem == val]
    return idxs

def filter_list(list, idxs):
    output = []
    for i, item in enumerate(list):
        if i not in idxs:
            output.append(item)
    return output

def get_rowwise_norm(embedding):
    sum = 0
    for row in embedding:
        norm = np.linalg.norm(row)
        sum += norm
    return sum

def save_nparr(embed_fn, emb):
    with open(embed_fn, 'w') as f:
        np.save(f, emb)

def _get_timestamp_str():
    time_str = time.strftime("%H%M_%S")
    date_str = time.strftime("%Y%m%d")
    return '{0}_{1}'.format(date_str, time_str)

def create_timestamped_dir(dir):
    timestamp_str = _get_timestamp_str()
    output_dir = os.path.join(dir, timestamp_str)
    os.makedirs(output_dir)
    return output_dir