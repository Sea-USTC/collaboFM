from collaboFM.configs.config import CN
from collaboFM.register import register_config


def extend_training_cfg(cfg):
    # ---------------------------------------------------------------------- #
    # Trainer related options
    # ---------------------------------------------------------------------- #
    cfg.trainer = CN()

    cfg.trainer.type = 'general'

    cfg.trainer.sam = CN()
    cfg.trainer.sam.adaptive = False
    cfg.trainer.sam.rho = 1.0
    cfg.trainer.sam.eta = .0

    cfg.train = CN()

    cfg.train.local_update_steps = 1
    cfg.train.batch_or_epoch = 'batch'
    cfg.train.batchsize = 128

    cfg.train.optimizer = CN(new_allowed=True)
    cfg.train.optimizer.type = 'SGD'
    cfg.train.optimizer.lr = 0.1
    cfg.train.optimizer.momentum = 0.9
    cfg.train.optimizer.weight_decay = 0.0


    # you can add new arguments 'aa' by `cfg.train.scheduler.aa = 'bb'`
    cfg.train.scheduler = CN(new_allowed=True)
    cfg.train.scheduler.type = ''
    cfg.train.scheduler.warmup_ratio = 0.0

    # ---------------------------------------------------------------------- #
    # Gradient related options
    # ---------------------------------------------------------------------- #
    cfg.grad = CN()
    cfg.grad.grad_clip = -1.0  # negative numbers indicate we do not clip grad
    cfg.grad.grad_accum_count = 1

    # ---------------------------------------------------------------------- #
    # Early stopping related options
    # ---------------------------------------------------------------------- #
    cfg.early_stop = CN()

    # patience (int): How long to wait after last time the monitored metric
    # improved.
    # Note that the actual_checking_round = patience * cfg.eval.freq
    # To disable the early stop, set the early_stop.patience to 0
    cfg.early_stop.patience = 5
    # delta (float): Minimum change in the monitored metric to indicate an
    # improvement.
    cfg.early_stop.delta = 0.0
    # Early stop when no improve to last `patience` round, in ['mean', 'best']
    cfg.early_stop.improve_indicator_mode = 'best'

    # --------------- register corresponding check function ----------
    cfg.register_cfg_check_fun(assert_training_cfg)


def assert_training_cfg(cfg):
    if cfg.train.batch_or_epoch not in ['batch', 'epoch']:
        raise ValueError(
            "Value of 'cfg.train.batch_or_epoch' must be chosen from ["
            "'batch', 'epoch'].")


register_config("fl_training", extend_training_cfg)