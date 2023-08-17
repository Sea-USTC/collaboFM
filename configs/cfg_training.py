from collaboFM.configs.config import CN
from collaboFM.register import register_config

import logging
logger = logging.getLogger(__name__)

def extend_training_cfg(cfg):
    # ---------------------------------------------------------------------- #
    # Trainer related options
    # ---------------------------------------------------------------------- #

    cfg.train = CN()

    cfg.train.local_update_steps = 1
    cfg.train.batchsize = 128

    cfg.train.optimizer = CN(new_allowed=True)
    cfg.train.optimizer.type = 'SGD'
    cfg.train.optimizer.lr = 0.1
    cfg.train.optimizer.momentum = 0.9
    cfg.train.optimizer.weight_decay = 0.0


    # you can add new arguments 'aa' by `cfg.train.scheduler.aa = 'bb'`

    # ---------------------------------------------------------------------- #
    # Gradient related options
    # ---------------------------------------------------------------------- #
    cfg.grad = CN()
    cfg.grad.grad_clip = -1.0  # negative numbers indicate we do not clip grad

    # ---------------------------------------------------------------------- #
    # Early stopping related options
    # ---------------------------------------------------------------------- #
    cfg.collabo_tqn=CN()
    cfg.collabo_tqn.key_train_round=20
    cfg.collabo_tqn.tau = 200
    cfg.collabo_tqn.mu=0.05
    cfg.collabo_tqn.epoch_list = [1,1,1]

    cfg.tqn_model=CN()
    cfg.tqn_model.layernum = 4
    cfg.tqn_model.selfatt = True
    cfg.tqn_model.weight_true = 0.5
    cfg.tqn_model.head_num = 16
    # --------------- register corresponding check function ----------
    cfg.register_cfg_check_fun(assert_training_cfg)


def assert_training_cfg(cfg):
    if cfg.train.batch_or_epoch not in ['batch', 'epoch']:
        raise ValueError(
            "Value of 'cfg.train.batch_or_epoch' must be chosen from ["
            "'batch', 'epoch'].")


register_config("fl_training", extend_training_cfg)