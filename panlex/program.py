import sys

import os

sys.path.insert(0, 'models')

from config_manager import ConfigManager

def main(config_files, output_folder):
    # Get config manager
    config_manager = ConfigManager(config_files=config_files,
                                   output_folder=output_folder)
    # Get logger
    logger = config_manager.logger

    train_model = config_manager.training_model
    train_model.run()


if __name__ == '__main__':
    os.nice(19)
    config_files = ['conf/default.conf']
    output_folder = 'output'    # Todo: valami permanens helyre
    main(config_files=config_files, output_folder=output_folder)