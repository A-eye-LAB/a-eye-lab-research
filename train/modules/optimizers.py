# Torch Imports
import torch.nn as nn
from torch.optim import Optimizer, Adam, SGD, Adadelta, Adagrad, AdamW, SparseAdam, Adamax, ASGD, LBFGS, RMSprop, Rprop

# ETC Imports
from typing import Any, Dict

# 옵티마이저 클래스 등록
OPTIMIZERS = {
    'Adam': Adam,
    'SGD': SGD,
    'Adadelta': Adadelta,
    'Adagrad': Adagrad,
    'AdamW': AdamW,
    'SparseAdam': SparseAdam,
    'Adamax': Adamax,
    'ASGD': ASGD,
    'LBFGS': LBFGS,
    'RMSprop': RMSprop,
    'Rprop': Rprop,
}

def get_optimizer(
        model: nn.Module, 
        optimizer_cfg: Dict[str, Any]
    ) -> Optimizer:
    
    optimizer_name = optimizer_cfg.get('NAME')
    
    if optimizer_name is None or optimizer_name not in OPTIMIZERS:
        supported_optimizers = ', '.join(OPTIMIZERS.keys())
        raise ValueError(
            f"Unsupported or unspecified optimizer: {optimizer_name}. "
            f"Supported optimizers are: {supported_optimizers}"
        )
    
    # optimizer 설정에서 NAME 키를 제외한 새로운 설정 딕셔너리 생성
    optimizer_params = {k: v for k, v in optimizer_cfg.items() if k != 'NAME'}
    
    wd_params, nwd_params = [], []
    for p in model.parameters():
        if p.dim() == 1:
            nwd_params.append(p)
        else:
            wd_params.append(p)
    
    params = [
        {"params": wd_params},
        {"params": nwd_params, "weight_decay": 0}
    ]

    optimizer_class = OPTIMIZERS[optimizer_name]
    
    return optimizer_class(params=params, **optimizer_params)

# 테스트 코드
if __name__ == "__main__":
    from torchvision import models
    from utils import load_yaml
    cfg = load_yaml('../configs/vit.yaml')

    model = models.resnet152()
    optimizer = get_optimizer(model, cfg['OPTIMIZER'])
    print(optimizer)