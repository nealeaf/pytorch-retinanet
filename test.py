import torch
import torchvision.transforms as transforms

from torch.autograd import Variable

from retinanet import RetinaNet
from encoder import DataEncoder
from PIL import Image, ImageDraw

import cv2


print('Loading model..')
net = RetinaNet()
net.load_state_dict(torch.load('checkpoint/ckpt.pth')['net'])
net.eval()
net.cuda()
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
])

print('Loading image..')
#img = Image.open('./image/000001.jpg')

cap = cv2.VideoCapture('Y:\\u_zhh\\data-jy\\视频\\8.mp4')
while True:
    ret, cv2img = cap.read() # 名称不能有汉字
    if not ret:
        break
    cv2img = cv2.cvtColor(cv2img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(cv2img)

    w = h = 600
    img = img.resize((w,h))

    print('Predicting..')
    x = transform(img)
    x = x.unsqueeze(0)
    x =x.cuda()

    draw = ImageDraw.Draw(img)

    encoder = DataEncoder()
    with torch.no_grad():
        loc_preds, cls_preds = net(x)

        print('Decoding..')
        boxes, labels, scores = encoder.decode(loc_preds.data, cls_preds.data, (w,h))

        for box in boxes:
            draw.rectangle(list(box), outline='red')
    img.show()
    img.close()
