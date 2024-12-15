# Copyright (c) Facebook, Inc. and its affiliates.

# Taken from: https://fairseq.readthedocs.io/en/latest/_modules/fairseq/optim/lr_scheduler/inverse_square_root_schedule.html
# to avoid pulling fairseq as a dependency

from torch.optim.lr_scheduler import LRScheduler

class InverseSquareRootSchedule(LRScheduler):
    """Decay the LR based on the inverse square root of the update number.

    We also support a warmup phase where we linearly increase the learning rate
    from some initial learning rate (``--warmup-init-lr``) until the configured
    learning rate (``--lr``). Thereafter we decay proportional to the number of
    updates, with a decay factor set to align with the configured learning rate.

    During warmup::

      lrs = torch.linspace(cfg.warmup_init_lr, cfg.lr, cfg.warmup_updates)
      lr = lrs[update_num]

    After warmup::

      decay_factor = cfg.lr * sqrt(cfg.warmup_updates)
      lr = decay_factor / sqrt(update_num)
    """
    def __init__(self, optimizer, warmup_updates=4000, warmup_init_lr=-1, lr=5e-5, last_epoch=-1):
        self.warmup_updates = warmup_updates
        self.warmup_init_lr = max(warmup_init_lr, 0)
        self.lr = lr
        self.warmup_end_lr = lr
        self.lr_step = (self.warmup_end_lr - self.warmup_init_lr) / self.warmup_updates
        self.decay_factor = self.warmup_end_lr * self.warmup_updates ** .5
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_updates:
            lr = self.warmup_init_lr + self.last_epoch * self.lr_step
        else:
            lr = self.decay_factor / (self.last_epoch ** .5)
        return [lr for _ in self.base_lrs]

    def step(self, epoch=None):
        if epoch == None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        lr = self.get_lr()[0]
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr
    
    def get_last_lr(self):
        return self.get_lr()
