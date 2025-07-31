import torch
import torch.onnx
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
import os
import sys
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)),".."))
from train.models import *
def export_to_onnx(model, input_shape, save_path):
    """PyTorch 모델을 ONNX로 변환"""
    model.eval()
    dummy_input = torch.randn(input_shape).cuda()

    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print(f"ONNX 모델이 {save_path}에 저장되었습니다.")

def quantize_onnx_model(onnx_path, quantized_path):
    """ONNX 모델 동적 양자화"""
    quantized_model = quantize_dynamic(
        onnx_path,
        quantized_path,
        weight_type=QuantType.QUInt8
    )
    print(f"양자화된 모델이 {quantized_path}에 저장되었습니다.")

def verify_onnx_model(onnx_path):
    """ONNX 모델 검증"""
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)
    print("ONNX 모델 검증이 완료되었습니다.")

if __name__ == "__main__":
    # 사용 예시
    model_path = "../weights/fastvit.pt"
    model = FastViT(num_classes=2, pretrained=False)  # 여기에 실제 PyTorch 모델을 넣으세요
    model.load_state_dict(torch.load(model_path, map_location="cuda", weights_only=True))
    model.to("cuda")
    model.eval()
    input_shape = (1, 3, 224, 224)  # 예시 입력 shape

    # ONNX 변환
    onnx_path = "./model_fastvit.onnx"
    export_to_onnx(model, input_shape, onnx_path)

    # 모델 검증
    verify_onnx_model(onnx_path)

    # 양자화
    quantized_path = "./model_fastvit_quantized.onnx"
    quantize_onnx_model(onnx_path, quantized_path)