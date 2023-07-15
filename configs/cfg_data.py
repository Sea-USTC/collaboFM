import logging

from configs.config import CN
from collaboFM.register import register_config

logger = logging.getLogger(__name__)


def extend_data_cfg(cfg):
    # ---------------------------------------------------------------------- #
    # Dataset related options
    # ---------------------------------------------------------------------- #
    cfg.data = CN()

    cfg.data.root = 'data'
    cfg.data.dataset = 'toy'
    cfg.data.load_all_dataset = True
    cfg.data.save_data = False  # whether to save the generated toy data
    cfg.data.args = []  # args for external dataset, eg. [{'download': True}]
    cfg.data.splitter = ''
    cfg.data.splitter_args = []  # args for splitter, eg. [{'alpha': 0.5}]

    cfg.data.transform = [
    ]  # transform for x, eg. [['ToTensor'], ['Normalize', {'mean': [
    # 0.9637], 'std': [0.1592]}]]
    cfg.data.target_transform = []  # target_transform for y, use as above

    # If not provided, use `cfg.data.transform` for all splits
    cfg.data.val_transform = []
    cfg.data.val_target_transform = []
    cfg.data.val_pre_transform = []
    cfg.data.test_transform = []
    cfg.data.test_target_transform = []
    cfg.data.test_pre_transform = []

    # data.file_path takes effect when data.type = 'files'
    cfg.data.file_path = ''


    # --------------- register corresponding check function ----------
    cfg.register_cfg_check_fun(assert_data_cfg)


def assert_data_cfg(cfg):
    pass


register_config("data", extend_data_cfg)