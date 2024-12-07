import torch
import onnxruntime as ort
import torchvision.transforms as transforms
from PIL import Image
from src.models.swin import Swin_Large

import time

def load_model(model_path, num_classes=2):
    """pt file 인스턴스 생성"""
    model = Swin_Large(num_classes=num_classes, pretrained=False)
    model.load_state_dict(torch.load(model_path))  # 모델 경로 수정
    model.eval()  # 평가 모드로 설정
    return model

def load_onnx_model(onnx_path):
    """onnx file 인스턴스 생성"""
    return ort.InferenceSession(onnx_path)

def preprocess_image(image_path):
    """이미지 전처리"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Swin Transformer 입력 크기
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet 기준 정규화
    ])

    # 이미지 로드 및 전처리
    image = Image.open(image_path).convert('RGB')  # 이미지 로드 및 RGB로 변환
    image = transform(image).unsqueeze(0)  # 배치 차원 추가
    return image

def predict_pt(model, image_tensor):
    """Prediction pt model"""
    with torch.no_grad():  # 그래디언트 계산 비활성화
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)  # 가장 높은 확률의 클래스 인덱스 추출
    return predicted.item(), output  # 예측된 클래스 반환

def predict_onnx(ort_session, image_tensor):
    """Prediction onnx model"""
    # 예측 수행
    inputs = {ort_session.get_inputs()[0].name: image_tensor}  # 입력 이름과 배열을 딕셔너리로 전달
    outputs = ort_session.run(None, inputs)
    return outputs[0]


def convert_to_onnx(model, onnx_path):
    """Convert script to onnx"""
    # 더미 입력 생성 (모델의 입력 크기에 맞게)
    dummy_input = torch.randn(1, 3, 224, 224)  # 배치 크기 1, RGB 이미지, 224x224 크기

    # ONNX로 변환
    torch.onnx.export(
        model,               # 변환할 모델
        dummy_input,        # 더미 입력
        onnx_path,          # 저장할 ONNX 파일 경로
        export_params=True, # 모델의 파라미터도 함께 저장
        opset_version=11,   # ONNX 버전 (11 이상 권장)
        do_constant_folding=True,  # 상수 폴딩 최적화
        input_names=['input'],      # 입력 이름
        output_names=['output'],     # 출력 이름
        dynamic_axes={'input': {0: 'batch_size'},  # 배치 크기 동적 설정
                      'output': {0: 'batch_size'}}
    )
    print(f"Model has been converted to ONNX and saved at {onnx_path}")

if __name__ == '__main__':
    # conver_script()
    image_path = 'src/onnx/cataract.png'
    model_pt_path = 'outputs/model.pt'
    model_onnx_path = 'test.onnx'


    model_pt = load_model(model_pt_path)
    model_onnx = load_onnx_model(model_onnx_path)
    image_tensor = preprocess_image(image_path)

    start = time.time()
    predict_pt_model = predict_pt(model_pt, image_tensor)
    print('*Predict script model')
    print(predict_pt_model[1])
    print(time.time() - start)
    start = time.time()
    predict_onnx_model = predict_onnx(model_onnx, image_tensor.cpu().numpy())
    print('*Predict onnx model')
    print(predict_onnx_model)
    print(time.time() - start)

    # print(predict_class)
    # convert_to_onnx(model_pt, 'test.onnx')
