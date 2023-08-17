from collaboFM.configs.config import CN
from collaboFM.register import register_config

import logging
logger = logging.getLogger(__name__)


def extend_data_cfg(cfg):
    # ---------------------------------------------------------------------- #
    # Dataset related options
    # ---------------------------------------------------------------------- #
    cfg.data = CN()

    cfg.data.root = '/mnt/workspace/colla_group/data/'
    cfg.data.dataset = 'cifar10'
    cfg.data.load_all_dataset = True
    cfg.data.splitter = ''
    cfg.data.splitter_args = []  # args for splitter, eg. [{'alpha': 0.5}]

    # --------------- register corresponding check function ----------
    cfg.register_cfg_check_fun(assert_data_cfg)


def assert_data_cfg(cfg):
    pass


register_config("data", extend_data_cfg)