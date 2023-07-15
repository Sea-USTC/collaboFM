import logging

from configs.config import CN
from collaboFM.register import register_config
import torch

logger = logging.getLogger(__name__)


def extend_fl_setting_cfg(cfg):
    # ---------------------------------------------------------------------- #
    # Federate learning related options
    # ---------------------------------------------------------------------- #
    cfg.federate = CN()

    cfg.federate.generic_fl_eval=False
    cfg.federate.use_hetero_model=False
    cfg.federate.client_num = 0
    cfg.federate.sample_client_num = -1
    cfg.federate.sample_mode="random"
    cfg.federate.total_round_num = 50
    cfg.federate.method = "FedAvg"
    cfg.federate.restore_from = ''
    cfg.federate.save_to = ''
    
    cfg.federate.client_resource={}   

    cfg.fm=CN()
    cfg.fm.use=False
    cfg.fm.llm="clip"
    # --------------- register corresponding check function ----------
    cfg.register_cfg_check_fun(assert_fl_setting_cfg)


def assert_fl_setting_cfg(cfg):
    # =============  client num related  ==============
    assert (cfg.federate.use_hetero_model and not cfg.federate.client_resource), "Client_resource need to be specified"



register_config("fl_setting", extend_fl_setting_cfg)