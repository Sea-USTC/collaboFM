from collaboFM.configs.config import CN
from collaboFM.register import register_config
import torch

import logging
logger = logging.getLogger(__name__)


def extend_fl_setting_cfg(cfg):
    # ---------------------------------------------------------------------- #
    # Federate learning related options
    # ---------------------------------------------------------------------- #
    cfg.federate = CN()

    cfg.federate.use_hetero_model=False
    cfg.federate.client_num = 0
    cfg.federate.sample_client_num = -1
    cfg.federate.total_round_num = 50
    cfg.federate.method = "FedAvg"
    
    cfg.client_resource=CN()
    cfg.client_resource.dataset=CN(new_allowed=True)
    cfg.client_resource.backbone=CN(new_allowed=True)
    cfg.client_resource.encoder_list=CN(new_allowed=True)
    cfg.client_resource.encoder_para_list=CN(new_allowed=True)
    cfg.client_resource.head_list=CN(new_allowed=True)
    cfg.client_resource.head_para_list=CN(new_allowed=True)

    cfg.fm=CN()
    cfg.fm.use=False
    # --------------- register corresponding check function ----------
    cfg.register_cfg_check_fun(assert_fl_setting_cfg)


def assert_fl_setting_cfg(cfg):
    # =============  client num related  ==============
    pass



register_config("fl_setting", extend_fl_setting_cfg)