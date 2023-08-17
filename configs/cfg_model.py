from collaboFM.configs.config import CN
from collaboFM.register import register_config

import logging
logger = logging.getLogger(__name__)

def extend_model_cfg(cfg):
    # ---------------------------------------------------------------------- #
    # Model related options
    # ---------------------------------------------------------------------- #
    cfg.model = CN()
    cfg.model.backbone = "ResNet18Cifar10"

    cfg.model.encoder_list=[]
    cfg.model.encoder_para_list=[]
    cfg.model.head_list=[]
    cfg.model.head_para_list=[]
    cfg.model.pretrained=False
    # ---------------------------------------------------------------------- #
    # Criterion related options
    # ---------------------------------------------------------------------- #

    # ---------------------------------------------------------------------- #
    # regularizer related options
    # ---------------------------------------------------------------------- #


    # --------------- register corresponding check function ----------
    cfg.register_cfg_check_fun(assert_model_cfg)


def assert_model_cfg(cfg):
    pass

register_config("model", extend_model_cfg)
