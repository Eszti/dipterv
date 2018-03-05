import sys

from data_model import DataModelWrapper
from continue_model import ContModel
from plot_model import PlotModel
from train_model import TrainModel
from test_model import TestModel
from valid_model import ValidModel

sys.path.insert(0, 'utils')
sys.path.insert(0, 'models')
from configparser import ConfigParser

import os
import logging
import config_models as config_models

import strings
from io_helper import copy_files, save_pickle
from logging_helper import configure_logging, str_to_loglevel, create_timestamped_dir


class ConfigManager:
    def __init__(self, config_files, output_folder):
        self.name = strings.CONFIG_MANAGER_NAME
        self.output_dir =  os.path.join(create_timestamped_dir(output_folder))
        self._get_config(config_files)
        # Init logger
        file_log_level = str_to_loglevel[self.cfg.get('logging', 'file_log_level').lower()]
        console_log_level = str_to_loglevel[self.cfg.get('logging', 'console_log_level').lower()]
        configure_logging(file_log_path=os.path.join(self.output_dir, 'log.txt'),
                          console_log_level=console_log_level,
                          file_log_level=file_log_level)
        self.logger = logging.getLogger()
        # Copy config files
        config_files_output_dir = os.path.join(self.output_dir, self.name)
        copy_files(output_dir=config_files_output_dir,
                   orig_files=config_files,
                   logger=self.logger)
        self.logger.debug('Config files are saved to {}'.format(config_files_output_dir))
        # Getting config models
        self.language_config = config_models.LanguageConfig(cfg=self.cfg)
        self.cont_config = config_models.ContConfig(cfg=self.cfg)
        self.data_wrapper_config = config_models.DataWrapperConfig(cfg=self.cfg)
        self.embedding_config = config_models.EmbeddingConfig(cfg=self.cfg)
        self.training_config = config_models.TrainingConfig(cfg=self.cfg)
        self.validation_config = config_models.ValidationConfig(cfg=self.cfg)
        self.test_config = config_models.TestConfig(cfg=self.cfg)
        # Getting models
        self.cont_model = ContModel(cont_config=self.cont_config)
        self.data_model_wrapper = DataModelWrapper(data_wrapper_config=self.data_wrapper_config,
                                                   embedding_config=self.embedding_config,
                                                   language_config=self.language_config)
        if strings.VALID in self.data_model_wrapper.data_models.keys():
            self.validation_model = ValidModel(valid_config=self.validation_config,
                                               data_model_wrapper=self.data_model_wrapper,
                                               language_config=self.language_config,
                                               output_dir=self.output_dir)
        else:
            self.validation_model = None
        self.plot_model = PlotModel(input_dir=self.output_dir)
        self.training_model = TrainModel(train_config=self.training_config,
                                         data_model_wrapper=self.data_model_wrapper,
                                         language_config=self.language_config,
                                         output_dir=self.output_dir,
                                         cont_model=self.cont_model,
                                         validation_model=self.validation_model,
                                         plot_model=self.plot_model)
        if strings.TEST in self.data_model_wrapper.data_models.keys():
            self.test_model = TestModel(test_config=self.test_config,
                                        data_model_wrapper=self.data_model_wrapper,
                                        language_config=self.language_config,
                                        output_dir=self.output_dir)
        else:
            self.test_model = None

    def _get_config(self, config_files_list):
        config_files = [fn for fn in config_files_list if fn is not None]
        not_found = [fn for fn in config_files if not os.path.exists(fn)]
        if not_found:
            raise Exception("Config file(s) not found: {0}".format(not_found))
        self.cfg = ConfigParser(os.environ)
        self.cfg.read(config_files)
        # Save new_cfg
        self._save_config()

    def _save_config(self):
        config_object_fn = os.path.join(self.output_dir, self.name, strings.CONFIG_OBJECT_FN)
        save_pickle(data=self.cfg,
                    filename=config_object_fn)