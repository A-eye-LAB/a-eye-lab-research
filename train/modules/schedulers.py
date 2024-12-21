# Torch Imports
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    StepLR,
    CosineAnnealingLR as TorchCosineAnnealingLR,
    LinearLR as TorchLinearLR,
    ReduceLROnPlateau    # 새로운 import 추가
)

# ETC Imports
from typing import List


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
        )
    }

    if scheduler_name not in schedulers:
        raise ValueError(
            f"Unavailable scheduler name >> {scheduler_name}. "
            f"Available schedulers: {list(schedulers.keys())}"
        )

    return schedulers[scheduler_name]()

