import torch.optim as Optim
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup as WarmupLinearSchedule


class WarmupOptimizer(object):
    def __init__(self, lr_base, optimizer, data_size, batch_size):
        self.optimizer = optimizer
        self._step = 0
        self.lr_base = lr_base
        self._rate = 0
        self.data_size = data_size
        self.batch_size = batch_size


    def step(self):
        self._step += 1

        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate

        self.optimizer.step()


    def zero_grad(self):
        self.optimizer.zero_grad()


    def rate(self, step=None):
        if step is None:
            step = self._step

        if step <= int(self.data_size / self.batch_size * 0.5):
            r = self.lr_base * 1/5.
        elif step <= int(self.data_size / self.batch_size * 1):
            r = self.lr_base * 2/5.
        elif step <= int(self.data_size / self.batch_size * 1.5):
            r = self.lr_base * 3/5.
        elif step <= int(self.data_size / self.batch_size * 2):
            r = self.lr_base * 4/5.
        elif step <= int(self.data_size / self.batch_size * 2.5):
            r = self.lr_base * 5/5.
        elif step <= int(self.data_size / self.batch_size * 3.0):
            r = self.lr_base * 4/5.
        elif step <= int(self.data_size / self.batch_size * 3.5):
            r = self.lr_base * 3/5.
        elif step <= int(self.data_size / self.batch_size * 4):
            r = self.lr_base * 3/5.
        else:
            r = self.lr_base * 2/5.

        return r


def get_optim(__C, model, data_size, lr_base=None):
    if lr_base is None:
        lr_base = __C.lr_base

    return WarmupOptimizer(
        lr_base,
        Optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=0,
            betas=__C.opt_betas,
            eps=__C.opt_eps
        ),
        data_size,
        __C.batch_size
    )

def get_Adam_optim(__C, model):
    return Optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=__C.lr_base,
        betas=__C.opt_betas,
        eps=__C.opt_eps
    )

def adjust_lr(optim, decay_r):
    optim.lr_base *= decay_r

def get_Adam_optim_v2(config, model):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.lr_base, weight_decay=0.01, correct_bias=False)
    scheduler = WarmupLinearSchedule(optimizer, num_training_steps=config.num_train_optimization_steps,
                                     num_warmup_steps=config.warmup_proportion * config.num_train_optimization_steps)
    return optimizer, scheduler
