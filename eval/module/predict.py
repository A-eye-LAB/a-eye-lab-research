import torch
from tqdm import tqdm
from scipy.special import softmax
import numpy as np
import onnxruntime
import os
import sys
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)),"../../"))
from train.models import *

def onnx_predict(model_path, dataloader):
    session = onnxruntime.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name 
    all_preds = []
    all_labels = []
    all_probs = []  # 확률값 저장을 위한 리스트

    for data, label in tqdm(dataloader, desc='Evaluating', unit='batch'):
        input_data = data.numpy()
        outputs = session.run(None, {input_name: input_data})
        
        # 클래스별 확률값
        probabilities = softmax(outputs[0], axis=1)
        preds = np.argmax(probabilities, axis=1)
        
        all_probs.extend(probabilities[:,1])
        all_preds.extend(preds)
        all_labels.extend(label.numpy())

    return all_labels, all_preds

def pytorch_predict(model_path, dataloader):

    model = FastViT(num_classes=2, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location="cuda", weights_only=True))
    model.to("cuda")
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data, label in tqdm(dataloader, desc='Evaluationg', unit='batch'):
            data = data.to("cuda")

            output = model(data)
            preds = torch.argmax(output, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(label.cpu().numpy())

    return all_labels, all_preds