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
    parser.add_argument('-o', dest='output_root_dir', required=False, help='Root dir for timestamped output dir')
    args = parser.parse_args()
    output_folder = 'output'
    output_root_dir = args.output_root_dir
    if output_root_dir is not None:
        output_folder = os.path.join(output_folder, output_root_dir)
    main(config_files=[args.config_file], output_folder=output_folder)