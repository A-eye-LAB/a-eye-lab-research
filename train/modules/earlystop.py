# Torch Imports
import torch
import torch.nn as nn

# ETC Imports
from typing import Optional

class EarlyStopping:
    def __init__(self, patience: int = 5, delta: float = 0.0, verbose: bool = False) -> None:
        self.patience: int = patience
        self.delta: float = delta
        self.verbose: bool = verbose
        self.counter: int = 0
        self.best_score: Optional[float] = None
        self.early_stop: bool = False
        self.val_loss_min: float = float('inf')

    def __call__(self, val_loss: float, model: nn.Module, path: str = 'checkpoint.pt') -> None:
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss: float, model: nn.Module, checkpoint_path: str) -> None:
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model to {checkpoint_path} ...')
        try:
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Model saved at {checkpoint_path}")
        except Exception as e:
            print(f"Error saving model: {e}")
        
        self.val_loss_min = val_loss
