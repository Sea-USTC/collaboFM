from collaboFM.configs.config import CN
from collaboFM.register import register_config
import logging
logger = logging.getLogger(__name__)

def extend_evaluation_cfg(cfg):

    # ---------------------------------------------------------------------- #
    # Evaluation related options
    # ---------------------------------------------------------------------- #
    cfg.eval = CN(
        new_allowed=True)  # allow user to add their settings under `cfg.eval`

    cfg.eval.freq = 1
    cfg.eval.batchsize = 128
    # --------------- register corresponding check function ----------
    cfg.register_cfg_check_fun(assert_evaluation_cfg)


def assert_evaluation_cfg(cfg):
    pass


register_config("eval", extend_evaluation_cfg)