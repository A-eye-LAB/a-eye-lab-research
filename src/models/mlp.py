import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class MLP_TEST(nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super(MLP_TEST, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.dropout_prob = 0.5 # 50%의 노드에 대한 가중치 계산을 하지 않기 위한 설정
        self.batch_norm1 = nn.BatchNorm1d(512) # 1dimension이기 때문에 BatchNorm1d를 사용함.
        self.batch_norm2 = nn.BatchNorm1d(256)


    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.fc1(x)
        x = self.batch_norm1(x)
        x = F.relu(x) # sigmoid(x)
        x = F.dropout(x, training=self.training, p=self.dropout_prob)
        x = self.fc2(x)
        x = self.batch_norm2(x)
        x = F.relu(x) # sigmoid(x)
        x = F.dropout(x, training=self.training, p=self.dropout_prob)
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x
    

if __name__ == '__main__':
    # 모델 생성 및 테스트
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(num_classes=10).to(device)
    
    # 모델 구조 확인
    print(model)