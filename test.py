import torch
import torchvision.transforms as transforms
from torchsummary import summary

from torch.autograd import Variable

from retinanet import RetinaNet
from encoder import DataEncoder
from PIL import Image, ImageDraw

import cv2
import time


print('Loading model..')
net = RetinaNet()
net.load_state_dict(torch.load('checkpoint/ckpt.pth')['net'])
net.eval()
net.cuda()
#summary(net, (3, 224, 224))

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
])

print('Loading image..')
#img = Image.open('./image/000001.jpg')

cap = cv2.VideoCapture('Z:\\u_zhh\\video\\jy2.mp4')
while True:
    ret, cv2img = cap.read() # 名称不能有汉字
    if not ret:
        break
    cv2img = cv2.cvtColor(cv2img, cv2.COLOR_BGR2RGB)

    s_time = time.clock()
    inputSize = 600
    h, w, _ = cv2img.shape
    scale = max(cv2img.shape[1], cv2img.shape[2])
    w = int(inputSize * w / scale)
    h = int(inputSize * h / scale)

    cv2img = cv2.resize(cv2img, (w, h))
    img = Image.fromarray(cv2img)

    #img = img.resize((w,h))

    print('Predicting..')
    x = transform(img)
    x = x.unsqueeze(0)
    x =x.cuda()

    #draw = ImageDraw.Draw(img)

    encoder = DataEncoder()
    with torch.no_grad():
        loc_preds, cls_preds = net(x)

        print('Decoding..')
        boxes, labels, scores = encoder.decode(loc_preds.data, cls_preds.data, (w,h))

        for box in boxes:
            cv2.rectangle(cv2img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
            #draw.rectangle(list(box), outline='red')
    e_time = time.clock()
    print('detect time:' + str(e_time - s_time))
    #img.show()
    cv2.imshow('detect', cv2img)
    cv2.waitKey(1)
