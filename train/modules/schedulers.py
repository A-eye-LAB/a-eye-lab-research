# Torch Imports
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    StepLR,
    CosineAnnealingLR as TorchCosineAnnealingLR,
    LinearLR as TorchLinearLR,
    ReduceLROnPlateau,    # 새로운 import 추가
    CosineAnnealingWarmRestarts
)

# ETC Imports
from typing import List
from torch.optim.lr_scheduler import _LRScheduler
import math


class CosineAnnealingWithWarmupLR(_LRScheduler):
    """선형 웜업이 있는 코사인 어닐링 스케줄러"""
    
    def __init__(self, optimizer, warmup_epochs, max_epochs, min_lr=0.0, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        super(CosineAnnealingWithWarmupLR, self).__init__(optimizer, last_epoch)
        
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # 웜업 단계: 0에서 기본 lr까지 선형 증가
            alpha = self.last_epoch / self.warmup_epochs
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            # 코사인 어닐링 단계
            progress = (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
            return [self.min_lr + (base_lr - self.min_lr) * cosine_factor for base_lr in self.base_lrs]


def get_scheduler(optimizer: Optimizer, scheduler_name: str, **kwargs):
    schedulers = {
        'StepLR': lambda: StepLR(
            optimizer, 
            step_size=kwargs.get('step_size', 10), 
            gamma=kwargs.get('gamma', 0.1)
        ),
        'CosineLR': lambda: TorchCosineAnnealingLR(
            optimizer, 
            T_max=kwargs.get('max_epochs'),
            eta_min=kwargs.get('min_lr', 0.0),
        ),
        'LinearLR': lambda: TorchLinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=kwargs.get('min_lr', 0.0),
            total_iters=kwargs.get('max_epochs')
        ),
        'ReduceLROnPlateau': lambda: ReduceLROnPlateau(
            optimizer,
            mode=kwargs.get('mode', 'min'),
            factor=kwargs.get('factor', 0.1),
            patience=kwargs.get('patience', 10),
            min_lr=kwargs.get('min_lr', 0.0),
            verbose=kwargs.get('verbose', False)
        ),
        'CosineWarmupLR': lambda: CosineAnnealingWithWarmupLR(
            optimizer,
            warmup_epochs=kwargs.get('warmup_epochs', 5),
            max_epochs=kwargs.get('max_epochs'),
            min_lr=kwargs.get('min_lr', 0.0)
        )
    }

    if scheduler_name not in schedulers:
        raise ValueError(
            f"Unavailable scheduler name >> {scheduler_name}. "
            f"Available schedulers: {list(schedulers.keys())}"
        )

    return schedulers[scheduler_name]()

