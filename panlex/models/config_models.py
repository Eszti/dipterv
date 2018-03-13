import sys

sys.path.insert(0, 'utils')
sys.path.insert(0, 'base')

from loggable import Loggable
import strings

class Configable(Loggable):
    def __init__(self, cfg):
        Loggable.__init__(self)
        self.config = cfg

    def _log_cfg(self, section, key, value):
        self.logger.debug('Conf param read: [{0}]: {1} - {2}'.format(section, key, value))

    def get(self, cfg_key, type=None, section=None):
        if section is None:
            section = self.name
        if type is None:
            value = self.config.get(section, cfg_key)
        elif type == 'int':
            value = self.config.getint(section, cfg_key)
        elif type == 'float':
            value = self.config.getfloat(section, cfg_key)
        elif type == 'int_inf':
            value_str = self.config.get(section, cfg_key)
            if value_str == 'inf':
                value = None
            else:
                value = int(value_str)
        elif type == 'boolean':
            value = self.config.getboolean(section, cfg_key)
        elif type == 'list':
            value = self.config.get(section, cfg_key).split('|')
        elif type == 'intlist':
            value = self.config.get(section, cfg_key).split('|')
            value = [int(x) for x in value]
        self._log_cfg(section, cfg_key, value)
        return value

    def get_optional(self, cfg_key, type=None, section=None, default=None):
        if section is None:
            section = self.name
        if self.config.has_option(section, cfg_key):
            return self.get(cfg_key, type, section)
        else:
            self.logger.debug('No option is found: {0} - {1}'.format(section, cfg_key))
            if default is not None:
                self.logger.debug('Default value is used: {}'.format(default))
                return default
            return None

class LanguageConfig(Configable):
    def __init__(self, cfg):
        Configable.__init__(self, cfg)
        self.config = cfg
        self.name = strings.LANGUAGE_CONFIG_NAME
        self.langs = self.get('langs', type='list')

class ContConfig(Configable):
    def __init__(self, cfg):
        Configable.__init__(self, cfg)
        self.config = cfg
        self.name = strings.CONT_CONFIG_NAME
        self.cont = self.get('continue', type='boolean')
        if self.cont:
            self.input_folder = self.get('input_folder')
            self.epoch = self.get_optional('epoch', 'int')

class EmbeddingConfig(Configable):
    def __init__(self, cfg, name=strings.EMBEDDING_CONFIG_MAME):
        Configable.__init__(self, cfg)
        self.config = cfg
        self.name = name
        self.langcode_type = self.get('langcode_type')
        self.sil2fb_path = self.get('sil2fb_path')
        self.path = self.get('path')
        self.limit = self.get_optional('limit', type='int')
        self.format = self.get('format')
        self.encoding = self.get_optional('encoding', default='utf-8')

class DataModelConfig(Configable):
    def __init__(self, cfg, typestr):
        Configable.__init__(self, cfg)
        self.config = cfg
        self.name = typestr
        self.dir = self.get('dir')
        self.header = self.get('header', type='boolean')
        self.idx1 = self.get('idx1', type='int')
        self.idx2 = self.get('idx2', type='int')
        self.emb = self.get_optional('emb')
        if self.emb is not None:
            self.embedding_config = EmbeddingConfig(cfg=cfg, name=self.emb)

class DataWrapperConfig(Configable):
    def __init__(self, cfg):
        Configable.__init__(self, cfg)
        self.config = cfg
        self.name = strings.DATA_WRAPPER_CONFIG_NAME
        self.types = self.get('types', type='list')
        self.data_configs = dict()
        for typestr in self.types:
            self.data_configs[typestr] = DataModelConfig(cfg, typestr)

class TrainingConfig(Configable):
    def __init__(self, cfg):
        Configable.__init__(self, cfg)
        self.config = cfg
        self.name = strings.TRAINING_CONFIG_NAME
        self.lr_base = self.get('lr_base', type='float')
        self.epochs = self.get('epochs', type='int')
        self.iters = self.get_optional('iters', type='int')
        self.svd_mode = self.get('svd_mode', type='int')
        self.svd_f = self.get('svd_f',  type='int')
        self.batch_size = self.get('batch_size', type='int')
        self.save_only_on_valid = self.get('save_only_on_valid',  type='boolean')

class ValidationConfig(Configable):
    def __init__(self, cfg):
        Configable.__init__(self, cfg)
        self.cfg = cfg
        self.name = strings.VALIDATION_CONFIG_NAME
        self.do_on = self.get('do_on', type='int')
        self.do_prec_calc = self.get('do_prec_calc', type='boolean')
        self.precs_to_calc = self.get_optional('precs_to_calc', type='intlist')
        self.calc_loss = self.get('calc_loss', type='boolean')
        self.calc_small_sing = self.get('calc_small_sing', type='boolean')
        self.limit = self.get('limit', type='float')

class TestConfig(Configable):
    def __init__(self, cfg):
        Configable.__init__(self, cfg)
        self.cfg = cfg
        self.name = strings.TEST_CONFIG_NAME
        self.do_prec_calc = self.get('do_prec_calc', type='boolean')
        self.precs_to_calc = self.get_optional('precs_to_calc', type='intlist')
        self.calc_loss = self.get('calc_loss', type='boolean')
        self.calc_small_sing = self.get('calc_small_sing', type='boolean')
        self.limit = self.get('limit', type='float')
        self.epochs = self.get('epochs', type='list')
        self.input_folder = self.get('input_folder')

