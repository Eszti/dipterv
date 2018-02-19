import argparse
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

    test_model = config_manager.test_model
    test_model.plot_progress()


if __name__ == '__main__':
    os.nice(19)
    parser = argparse.ArgumentParser(description='Run dipterv.')
    parser.add_argument('-cf', dest='config_file', type=str, help='config file for the experiments')
    args = parser.parse_args()
    output_folder = 'output'        # Todo: valami permanens helyre
    main(config_files=[args.config_file], output_folder=output_folder)