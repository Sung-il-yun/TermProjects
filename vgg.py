import torch
from torch import optim
import torchvision
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.utils as vutils
import torchvision.transforms as transforms
from torch.autograd import Variable

from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import argparse
import imutils
import dlib
import cv2
import os

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.cnn = nn.Sequential(
            # 3 x 128 x 128
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # 64 x 64 x 64
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # 128 x 32 x 32
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # 256 x 16 x 16
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # 512 x 8 x 8
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
            # 512 x 4 x 4
        )
        self.fc = nn.Sequential(
            nn.Linear(512 * 4 * 4, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(2048, 30)
        )

    def forward(self, x):
        output = self.cnn(x)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        return output


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:  # Conv weight init
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:  # BatchNorm weight init
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)



model = VGG()
model.apply(weights_init)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00002)

model.eval()

model.load_state_dict(torch.load('train.pth', map_location=lambda storage, loc: storage))

classes = (
'클로이 모레츠', '크리스 헴스워스', '고현정', '한혜진', '한예슬', '홍진영', '전지현', '김태희', '마동석', '박나래', '스칼렛 요한슨', '배수지', '톰 크루즈', '톰 히들스턴',
'윌 스미스', '조지 클루니', '하정우', '장동건', '조인성', '전현무', '강호동', '김흥국', '이병헌', '이정재', '맷 데이먼', '박시후', '박수홍', '유아인', '유승준',
'실베스터 스탤론')

sm = torch.nn.Softmax(dim=1)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
fa = FaceAligner(predictor, desiredFaceWidth=128)

def test(PATH):
    image = cv2.imread(PATH)  # 이미지 경로 넣는 곳
    image = imutils.resize(image, width=800)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 2)

    (x, y, w, h) = rect_to_bb(rects[0])
    faceOrig = imutils.resize(image[y:y + h, x:x + w], width=128)
    faceAligned = fa.align(image, gray, rects[0])


    image = transform(faceAligned)
    image.unsqueeze_(0)
    outputs = model(image)
    probability = sm(outputs)
    _, predicted = torch.max(outputs, 1)

    #print("테스트: ", classes[predicted.tolist()[0]], 100 * probability[0][predicted.tolist()[0]].tolist(), '%')
    return classes[predicted.tolist()[0]], 100 * probability[0][predicted.tolist()[0]].tolist()