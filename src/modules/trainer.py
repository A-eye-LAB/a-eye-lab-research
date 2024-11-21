# Torch Imports
import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast

# Module Imports
from modules.metrics import evaluate_model, print_evaluation
from modules.earlystop import EarlyStopping

# ETC Imports
from tqdm import tqdm
from typing import Tuple, Dict, Optional, Any

class Trainer:
    def __init__(
            self, 
            model: nn.Module,                           # 모델 인스턴스
            criterion: nn.Module,                       # 손실 함수
            optimizer: optim.Optimizer,                 # 최적화 알고리즘
            scheduler: optim.lr_scheduler._LRScheduler, # 학습률 스케줄러
            scaler: GradScaler,                         # Mixed precision scaler
            config: Dict[str, Any],                     # 설정 값 (dict 등)
            device: torch.device,                       # 사용되는 장치 (CPU, GPU 등)
            checkpoint_dir: str,                        # 체크포인트 저장 디렉토리 경로
            train_loader: DataLoader,                   # 훈련 데이터 로더
            valid_loader: Optional[DataLoader] = None,  # 검증 데이터 로더 (옵션)
        ) -> None:

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.config = config
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.train_loader = train_loader
        self.valid_loader = valid_loader
    
    def train(self) -> None:
        """
        Full training logic
        """
        # EarlyStopping 객체 초기화
        early_stopping = EarlyStopping(
            patience=self.config['TRAIN']['PATIENCE'], 
            verbose=True
        ) if self.config["TRAIN"]["PATIENCE"] > 0 else None

        for epoch in range(self.config["TRAIN"]["EPOCHS"]):
            try:
                result = self._train_epoch(epoch, early_stopping)
                # Early Stopping 체크
                if result == False:
                    break
            except Exception as e:
                raise

    def _train_epoch(
            self, 
            epoch: int, 
            early_stopping: Optional[EarlyStopping]
        ) -> bool:

        self.model.train()
        total_loss = 0.0
        LR = self.scheduler.get_last_lr()[0]
        pbar = tqdm(self.train_loader, desc=f"Train Epoch [{epoch+1}/{self.config['TRAIN']['EPOCHS']}]")
        
        for data, target in pbar:
            # Forward pass 로직을 직접 구현
            data = data.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad()
            
            with autocast(enabled=self.config["TRAIN"]["AMP"], device_type=self.device.type):
                output = self.model(data)
                loss = self.criterion(output, target)

            # Grad Scale
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
                    
            total_loss += loss.item()
            train_loss = total_loss / len(self.train_loader)

            pbar.set_postfix(Loss=total_loss / len(self.train_loader), LR=LR)

        train_loss = total_loss / len(self.train_loader)
        
        # Validation 단계 실행
        valid_loss, val_metrics = self._valid_epoch()
        print_evaluation("validate", epoch, valid_loss, val_metrics)


        model_dir = self.checkpoint_dir / 'weights'
        model_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = model_dir / f'checkpoint_epoch_{epoch}.pt'

        if early_stopping is not None:
            # Early Stopping 체크
            early_stopping(valid_loss, self.model, path=checkpoint_path)
            if early_stopping.early_stop:
                return False

        self.scheduler.step(valid_loss)

        return True

    @torch.no_grad()
    def _valid_epoch(self) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        y_true = []
        y_pred = []

        for data, target in tqdm(self.valid_loader, desc="validation", leave=False):
            data = data.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)
            
            with autocast(enabled=self.config["TRAIN"]["AMP"], device_type=self.device.type):
                output = self.model(data)
                loss = self.criterion(output, target)

            total_loss += loss.item()
            _, predicted = torch.max(output, 1)

            y_true.extend(target.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
        
        valid_loss = total_loss / len(self.valid_loader)
        metrics = evaluate_model(y_true, y_pred, self.config)

        return valid_loss, metrics
    
