import sys

sys.path.insert(0, 'models')

from config_manager import ConfigManager

def main(config_files, output_folder):
    # Get config manager
    config_manager = ConfigManager(config_files=config_files,
                                   output_folder=output_folder)
    # Get logger
    logger = config_manager.logger


if __name__ == '__main__':
    config_files = ['conf/default.conf']
    output_folder = 'output'    # Todo: valami permanens helyre
    main(config_files=config_files, output_folder=output_folder)