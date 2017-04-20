import os

class BaseConfig:
    def read_codes(self, cfg):
        section = 'codes'
        silcodes_dir = cfg.get(section, 'silcodes_dir')
        silcodes_type = cfg.get(section, 'silcodes')
        return silcodes_dir, silcodes_type

    def read_embedding(self, cfg):
        section = 'embedding'
        eng_emb = cfg.get(section, 'eng_emb_fn')
        self.emb_dir = cfg.get(section, 'emb_dir')
        self.eng_emb_fn = os.path.join(self.emb_dir, eng_emb)

    def read_swadesh(self, cfg):
        section = 'swadesh'
        swad_root_dir = cfg.get(section, 'swad_root_dir')
        num = cfg.getint(section, 'num')
        return swad_root_dir, num

    def read_train(self, cfg):
        section = 'train'
        self.verbose = cfg.getboolean(section, 'verbose')
        self.debug = cfg.getboolean(section, 'debug')
        self.log_freq = cfg.getint(section, 'log_freq')
        self.num_steps = cfg.getint(section, 'num_steps')
        self.learning_rate = cfg.getfloat(section, 'learning_rate')
        self.end_cond = cfg.getfloat(section, 'end_cond')
        self.max_iter = cfg.getint(section, 'max_iter')

class ConfigTrain(BaseConfig):
    def __init__(self, cfg):
        # Read general
        section = 'general'
        self.output_dir = cfg.get(section, 'output_dir')
        # Read codes
        silcodes_dir, silcodes_type = self.read_codes(cfg)
        # Read embedding
        self.read_embedding(cfg)
        # Read swadesh
        swad_root_dir, num = self.read_swadesh(cfg)
        # Read train
        self.read_train(cfg)

        # Put together paths according to num
        silcodes = silcodes_type + str(num) + '.json'   # filename without dir
        self.silcodes_fn = os.path.join(silcodes_dir, silcodes)
        self.swad_dir = os.path.join(swad_root_dir, 'swadesh{}'.format(str(num)))

class ConfigTranslate(BaseConfig):
    def __init__(self, cfg):
        # Read general
        section = 'general'
        self.output_root = cfg.get(section, 'output_dir')
        # Read codes
        section = 'codes'
        silcodes_dir, silcodes_type = self.read_codes(cfg)
        self.sil2fb_map = cfg.get(section, 'sil2fb_map')
        # Read embedding
        self.read_embedding(cfg)
        # Read swadesh
        section = 'swadesh'
        swad_root_dir, num = self.read_swadesh(cfg)
        eng_swad = cfg.get(section, 'eng_swad')
        # Read train
        self.read_train(cfg)

        # Put together paths according to num
        silcodes = silcodes_type + str(num) + '.json'   # filename without dir
        self.silcodes_fn = os.path.join(silcodes_dir, silcodes)
        self.swad_dir = os.path.join(swad_root_dir, 'swadesh{}'.format(str(num)))
        self.eng_swad_fn = os.path.join(self.swad_dir, eng_swad)
