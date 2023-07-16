from collaboFM.configs.config import CN
from collaboFM.register import register_config

import logging
logger = logging.getLogger(__name__)

def extend_model_cfg(cfg):
    # ---------------------------------------------------------------------- #
    # Model related options
    # ---------------------------------------------------------------------- #
    cfg.model = CN()
    cfg.model.backbone = "ResNetCifar10"
    cfg.model.hidden = 256
    cfg.model.dropout = 0.5
    cfg.model.in_channels = 0  # If 0, model will be built by data.shape
    cfg.model.out_channels = 1
    cfg.model.input_shape = ()  # A tuple, e.g., (in_channel, h, w)

    cfg.model.encoder_list=[]
    cfg.model.encoder_para_list=CN(new_allowed=True)
    cfg.model.head_list=[]
    cfg.model.head_para_list=CN(new_allowed=True)
    # ---------------------------------------------------------------------- #
    # Criterion related options
    # ---------------------------------------------------------------------- #
    cfg.criterion = CN()

    cfg.criterion.type = 'cross_entropy'

    # ---------------------------------------------------------------------- #
    # regularizer related options
    # ---------------------------------------------------------------------- #
    cfg.regularizer = CN()

    cfg.regularizer.type = ''
    cfg.regularizer.mu = 0.

    # --------------- register corresponding check function ----------
    cfg.register_cfg_check_fun(assert_model_cfg)


def assert_model_cfg(cfg):
    pass

register_config("model", extend_model_cfg)
