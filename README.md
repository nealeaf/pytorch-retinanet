# PyTorch-RetinaNet

Train _RetinaNet_ with _Focal Loss_ in PyTorch.

## Installation

1) Clone this repo

```
git clone https://github.com/nealeaf/pytorch-retinanet.git
```

2) Install the required packages:

```
conda install pytorch torchvision cudatoolkit=10.1
```
## Train

1) download the pretrain model by follow url
```
resnet18: https://download.pytorch.org/models/resnet18-5c106cde.pth
resnet34: https://download.pytorch.org/models/resnet34-333f7ec4.pth
resnet50: https://download.pytorch.org/models/resnet50-19c8e357.pth
resnet101: https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
resnet152: https://download.pytorch.org/models/resnet152-b121ed2d.pth
```

2) convert model by script get_state_dict.py
```
python script/get_state_dict.py
```

3) Prepare your data in coco format, config train.txt, val.txt

4) run script train.py

```
python train.py --lr 0.0001
```
5) resume train

```
python train.py --lr 0.0001 --resume
```

6) view training loss. run server, and browse http://localhost:8097

```
python -m visdom.server
```
## Test
1) run script test.py

```
python test.py
```



Reference:  
[1] [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)  
[2] [kuangliu/pytorch-retinanet](https://github.com/kuangliu/pytorch-retinanet.git)