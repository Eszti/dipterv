import time

import os

def _get_timestamp_str():
    time_str = time.strftime("%H%M_%S")
    date_str = time.strftime("%Y%m%d")
    return '{0}_{1}'.format(date_str, time_str)

def create_timestamped_dir(dir):
    timestamp_str = _get_timestamp_str()
    output_dir = os.path.join(dir, timestamp_str)
    os.makedirs(output_dir)
    return output_dir
