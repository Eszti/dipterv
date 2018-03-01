import argparse
from configparser import ConfigParser

import os
import itertools
import stat
import re

descriptor_fn = 'descriptor.conf'
sample_conf_fn = 'default.conf.sample'

def _read_list_values(conf, section, cfg_key):
    value = conf.get(section, cfg_key).split(';')
    if '' in value:
        value.remove('')
    return value

def main(dir):
    # Get output root folder
    output_root_folder = os.path.basename(os.path.normpath(dir))

    # Load config object
    descriptor_path = os.path.join(dir, descriptor_fn)
    descriptor_conf = ConfigParser(os.environ)
    descriptor_conf.read(descriptor_path)

    # Checking whether needed files exist
    execute_dir = descriptor_conf.get('general', 'execute_dir')

    # Read params and their values from the descriptor file
    params = _read_list_values(conf=descriptor_conf, section='params', cfg_key='params')
    list_of_param_lists = []
    for param in params:
        param_list = _read_list_values(conf=descriptor_conf, section='params', cfg_key=param)
        list_of_param_lists.append(param_list)

    # Creating combinations for params
    combinations = []
    i = 0
    for element in itertools.product(*list_of_param_lists):
        print('{}:\t{}'.format(i,element))
        combinations.append(element)
        i += 1

    # Read the content of the sample config file
    with open(os.path.join(dir, sample_conf_fn)) as f:
        sample_conf_str = f.read()

    # Create dir for config files
    conf_dir = os.path.join(dir, 'conf')
    if not os.path.exists(conf_dir):
        os.makedirs(conf_dir)

    # Create config files
    for i, combination in enumerate(combinations):
        new_config_string = sample_conf_str
        new_conf_fn = os.path.join(conf_dir, '{}.conf'.format(i))
        for i, val in enumerate(combination):
            original_pattern = '\n{}:.*\n'.format(params[i])
            replace_pattern = '\n{}: {}\n'.format(params[i], val)
            new_config_string = re.sub(pattern=original_pattern, repl=replace_pattern, string=new_config_string)
        with open(new_conf_fn, 'w') as f:
            f.writelines(new_config_string)

    bash_string = ''
    # Create bash script for running
    bash_string = '#!/bin/bash\n\n' \
                  '(cd {0} &&\n\n' \
                  'for conf_fn in {1}/*; ' \
                  'do python {2} ' \
                  '-cf $conf_fn ' \
                  '-o {3}; done;)'\
        .format(execute_dir,
                conf_dir,
                os.path.join(execute_dir, 'program.py'),
                output_root_folder)

    if bash_string != '':
        bash_fn = os.path.join(dir, 'run_test.sh')
        with open(bash_fn, 'w') as f:
            f.writelines(bash_string)
        st = os.stat(bash_fn)
        os.chmod(bash_fn, st.st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=
                                     'Generates config files according to a given descriptor and sample configs.')
    parser.add_argument('-d', '--dir', type=str, required=True,
                        help='Directory containing a descriptor file and a sample config')
    args = parser.parse_args()

    main(dir=args.dir)