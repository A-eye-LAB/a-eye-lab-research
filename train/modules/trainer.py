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

import os
import shutil
from pathlib import Path
import wandb
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
            wandb : wandb,                              # WandB
            checkpoint_dir: str,                        # 체크포인트 저장 디렉토리 경로
            fold_idx: int,                               # 폴드 인덱스
            train_loader: DataLoader,                   # 훈련 데이터 로더
            valid_loader: Optional[DataLoader] = None,  # 검증 데이터 로더 (옵션)
            test_loader: Optional[DataLoader] = None,   # 테스트 데이터 로더 (옵션)
            resume: bool = False,
        ) -> None:

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.config = config
        self.device = device
        self.wandb = wandb
        self.checkpoint_dir = checkpoint_dir
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.fold_idx = fold_idx
        self.model_dir = self.checkpoint_dir / 'weights'
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.resume = resume

    def train(self) -> None:
        """
        Full training logic
        """
        # EarlyStopping 객체 초기화
        early_stopping = EarlyStopping(
            patience=self.config['TRAIN']['PATIENCE'],
            verbose=True
        ) if self.config["TRAIN"]["PATIENCE"] > 0 else None

        st_epoch = 0

        if self.resume:
            save_dir = self.config["SAVE_DIR"]

            try:
                weight_dir = [p for p in os.listdir(save_dir) if type(self.model).__name__ == "_".join(p.split("_")[:-2])]
                weight_dir = save_dir / Path(weight_dir[-2]) / 'weights'

                weight_list = os.listdir(weight_dir)
                def get_epoch(dir:str):
                    return int(dir.split("_")[-1][:-3])
                st_epoch = sorted(list(map(get_epoch, weight_list)))[-1] + 1

                self.model.load_state_dict(torch.load(weight_dir/f'checkpoint_epoch_{st_epoch-1}.pt', weights_only=True))
                self.scheduler.last_epoch = st_epoch

            except IndexError:
                print("\n===============================================================")
                print("Model reload error: Check the latest directory of the model")
                print(f"{type(self.model).__name__} within {save_dir}")
                print("===============================================================")
                shutil.rmtree(self.checkpoint_dir)
                return False

        for epoch in range(st_epoch, self.config["TRAIN"]["EPOCHS"]):
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

            pbar.set_postfix(Loss=train_loss / len(self.train_loader), LR=LR)

        # Validation 단계 실행
        valid_loss, val_metrics = self._valid_epoch()
        print_evaluation("validate", epoch, valid_loss, val_metrics)

        # Test 단계 실행
        test_loss, test_metrics = self.test_epoch()
        print_evaluation("test", epoch, test_loss, test_metrics)

        self.wandb.log({
            "fold_idx": self.fold_idx,
            "epoch": epoch,
            "learning_rate": LR,
            "train/train_loss": train_loss,
            "valid/valid_loss": valid_loss,
            **val_metrics  # validation 로그 추가
        })

        checkpoint_path = self.model_dir / f'checkpoint_epoch_{epoch}.pt'

        if early_stopping is not None:
            # Early Stopping 체크
            early_stopping(test_loss, self.model, path=checkpoint_path)
            if early_stopping.early_stop:
                return False
        else:
            try:
                torch.save(self.model.state_dict(), checkpoint_path)
                print(f"Model saved at {checkpoint_path}")
            except Exception as e:
                print(f"Error saving model: {e}")
            if epoch != 0:
                os.remove(self.model_dir/f'checkpoint_epoch_{epoch-1}.pt')

        self.scheduler.step()

        return True

    @torch.no_grad()
    def _valid_epoch(self) -> Tuple[float, Dict[str, float]]:
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
    
    @torch.no_grad()
    def test_epoch(self) -> Tuple[float, Dict[str, float]]:
        self.model.eval()
        total_loss = 0.0
        y_true = []
        y_pred = []

        for data, target in tqdm(self.test_loader, desc="test", leave=False):
            data = data.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)

            with autocast(enabled=self.config["TRAIN"]["AMP"], device_type=self.device.type):
                output = self.model(data)
                loss = self.criterion(output, target)

            total_loss += loss.item()
            _, predicted = torch.max(output, 1)

            y_true.extend(target.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

        test_loss = total_loss / len(self.test_loader)
        metrics = evaluate_model(y_true, y_pred, self.config)

        return test_loss, metrics