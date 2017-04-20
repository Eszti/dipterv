import json
import numpy as np
from sklearn.preprocessing import normalize

learning_rates = [0.3, 0.1, 0.03, 0.01, 0.003, 0.001]

default_cfg_file = 'conf/default_train.conf'

default_cfg_content = ""
with open(default_cfg_file) as f:
    default_cfg_content = f.read()

for lr in learning_rates:
    new_content = default_cfg_content.replace('learning_rate: 1', 'learning_rate: {}'.format(lr))
    new_content = new_content.replace('output_dir: /home/eszti/projects/dipterv/univ_embedding/output',
                        'output_dir: /home/eszti/projects/dipterv/univ_embedding/output/{}'.format(lr))
    save_name = 'conf/train/train_conf_{}'.format(lr)
    print(save_name)
    with open(save_name, 'w') as f:
        f.write(new_content)