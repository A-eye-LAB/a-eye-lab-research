import torch.nn as nn
import torchvision.models as models
import pickle
import torch
import torch.onnx
from dataclasses import dataclass
from typing import Tuple, Optional

# 상수 정의
@dataclass
class ModelConfig:
    threshold: float = 0.65
    input_size: Tuple[int, int] = (224, 224)
    num_classes: int = 2
    pretrained: bool = True

class MobileNet_V3_Large(nn.Module):
    def __init__(self, config: ModelConfig):
        super(MobileNet_V3_Large, self).__init__()
        self.threshold = config.threshold
        
        with open(EMBEDDING_PATH, 'rb') as f:
            self.mean_embedding = pickle.load(f)

        self.model = models.mobilenet_v3_large(
            weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1
        )
        self.model.classifier = torch.nn.Identity()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        embedding = self.model(x).squeeze()
        mean_embedding = torch.tensor(self.mean_embedding, device=x.device)
        similarity = 1 - torch.nn.functional.cosine_similarity(
            embedding.unsqueeze(0), 
            mean_embedding.unsqueeze(0)
        )
        result = (similarity > self.threshold).float()
        return result, similarity

def create_model(config: Optional[ModelConfig] = None) -> MobileNet_V3_Large:
    """모델 인스턴스를 생성하는 팩토리 함수"""
    if config is None:
        config = ModelConfig()
    model = MobileNet_V3_Large(config)
    model.eval()
    return model

def export_to_onnx(model: nn.Module, config: ModelConfig) -> None:
    """모델을 ONNX 형식으로 내보내는 함수"""
    dummy_input = torch.randn(1, 3, *config.input_size)
    
    torch.onnx.export(
        model,
        dummy_input,
        ONNX_OUTPUT_PATH,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['result', 'similarity'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'result': {0: 'batch_size'},
            'similarity': {0: 'batch_size'}
        }
    )
    print("ONNX 모델이 성공적으로 저장되었습니다.")

if __name__ == "__main__":

    EMBEDDING_PATH = "/workspace/a-eye-lab-research/notebook/eye_detection/embedding_pkl/mean_embedding_tuning.pkl"
    ONNX_OUTPUT_PATH = "/workspace/a-eye-lab-research/export/output/eye_detection.onnx"
    config = ModelConfig()
    model = create_model(config)
    export_to_onnx(model, config)
