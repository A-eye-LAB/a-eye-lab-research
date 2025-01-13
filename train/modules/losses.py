import torch
import torch.nn as nn
import torch.nn.functional as F



def get_loss(loss_name: str, **kwargs) -> nn.Module:
    loss_functions = {
        'CrossEntropyLoss': lambda: nn.CrossEntropyLoss(**kwargs),
        'BCEWithLogitsLoss': lambda: nn.BCEWithLogitsLoss(**kwargs)
    }

    if loss_name not in loss_functions:
        raise ValueError(
            f"Unavailable loss function name >> {loss_name}. "
            f"Available loss functions: {list(loss_functions.keys())}"
        )

    return loss_functions[loss_name]()

