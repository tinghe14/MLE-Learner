From: 小土堆 Pytorch

Course Content
- [Intro-others](#IO)
- [Dataset类](#dataset)
- [TensorBoard使用](#TB)
- [Transforms使用](#TF)
- [DataLoader使用](#DL)
- [nn.Module基本骨架](#nnM)
- [卷积层](#卷积)
- 最大池化
- 非线形激活
- 线性层及其他层
- sequential
- 损失函数和反向传播
- 优化器
- 现有模型的使用
- 网络模型的保存与读取
- 完整模型训练套路
- 利用GPU的训练
- 完整模型验证套路
- 完结-看开源项目

## Intro-others
<a id='IO'></a>
- package的法宝： dir(): 打开，看见里面有什么； help():这个函数的工具书
- os package: 做两个路径的连接
- self: 一个函数的变量不能传递给另外一个函数，self可以把这个指定的函数给后面的使用
- pycharm里查看函数具体信息的方式： help(函数名) 或者 常按control+鼠标移到这个函数的位置+点击蓝色链接进入官方doc
~~~
import os 
root_dir = 'dataset/train'
label_dir = 'ants'
path = os.path.join(root_dir, label_dir) #dataset/train/ants
~~~

## Dataset类
<a id='dataset'></a>
- 两个重要类：Dataset, Dataloader
- Dataset作用：提供一种方式去获取数据及其label，并且有编号
  - 有的功能：
  - 如何获取每一个数据及其label
  - 告诉我么你总共有多少个数据（才知道什么时候这个数据集run wan了可以迭代下一次）
~~~
from torch.utils.data import Dataset
class Mydata(Dataset):
  def __init__(self, root_dir, label_dir):
    self.root_dir = root_dir
    self.label_dir = label_dir
    self.path = os.path.join(self.root_dir, self.label_dir)
    self.img_path = os.listdir(self.path) #获得图片所有的地址
    
  def __getitem__(self, idx):
    img_name = self.img_path[idx]
    img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
    img = PIL.Image.open(img_item_path)
    label = self.label_dir
    return img, label
    
  def __len__(self):
    return len(self.img_path)
~~~
- Dataloader作用：为后面的网络提供不同的数据形式，进行打包

## TensorBoard使用
<a id='TB'></a>
- SummaryWriter类的使用
  - add_scalar()的使用：显示train loss的方式
~~~
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('logs') #event建立在logs folder下面
for i in range(100):
    writer.add_scalar('y=2x', 3*i, i) #标题，x轴, y轴
writer.close()
>>> tensorboard --logdir=logs --port=6007 #terminal的输入 指定port
~~~
  - add_image()的使用：常用来观察训练结果
~~~
from torch.utils.tensorboard import SummaryWriter
import numpy as np 
from PIL import Image

writer = SummaryWriter('logs')
image_path = 'data/train/ants_image/01.jpg'
img_PIL = Image.open(image_path)
img_array = np.array(img_PIL) #图片输入的类型 和维度的位置 室友要求的
writer.add_image("test", img_array, 1, dataformats='HWC') # step, 如果想要在同一个画布 但是表示每一步不同的图片 这边改成2就行了
# 单独显示的话 改一个命名就行了
writer.close()
~~~

## Transforms使用
<a id='TF'></a>
- transformers主要是对图片进行处理
- transformer结构和用法(点开structure可以看到他的class)
  - 常见的class
    - Compose
    - ToTensor
    - ToPILImage
    - Normalize
    - Resize
    - Compose
  - 注意输入输出类型，多看官方文档
~~~
from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWritter('logs')
img_path = 'a/b/c.jpg'
img = Image.open(img_path)

# tensor数据类型
# 通过transforms.ToTensor去看两个问题
# 1. transforms如何使用
tensor_trans = transforms.ToTensor() #先实例化
tensor_img = tensor_trans(img)

writer.add_image('ToTensor', img_tensor)
write.close()
# 2. 为什么需要tensor数据类型
有各种反向传播需要的属性
~~~
- 合起来
~~~
import torchvision
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

train_set = torchvision.datasets.CIFAR10(root='./dataset', train=True, transform=dataset_transform, download=True)
test_set = torchvision.datasets.CIFAR10(root='./dataset', train=False, transform=dataset_transform, download=True)

writer = SummaryWriter('p10')
for i in range(10):
    img, target = test_set[i]
    writer.add_image('test_set', img, i)
~~~

## DataLoader使用
<a id='DL'></a>
- dataset: 查看数据
- dataloader: 加载器，从dataset怎么取，每次取多少(batch_size)就是由dataloader决定
  - num_workers: 多进程
~~~
import torchvision
from torch.utils.data import DataLoader

test_data = torchvision.datasets.CIFAR10(root='./dataset', train=False, transform=torchvision.transforms.ToTensor(), download=True)

test_loader = DataLoader(dataset=test_data, batch_size=4, shuffle=True, num_workers=0, drop_last=False)
# 测试数据集第一张图片
img, target = test_data[0]
print(img.shape) #torch.Size([3, 32, 32])
print(target) #3

writer = SummaryWriter('dataloader')
for epoch in range(2): #遍历两轮，且因shuffle=True, 意味着第一次和第二次是顺序不一样的图片
  step = 0
  for data in test_loader:
      imgs, targets = data
      print(imgs.shape) #torch.Size([4, 3, 32, 32]) 4张图片
      print(targets) #tensor([0,0,1,0])
      writer.add_images('Epoch: {}'.format(epoch), imgs, step)
      step += 1
    
writer.close()    
~~~

## nn.Module基本骨架
<a id='nnM'></a>
torch.nn vs torch.nnfunctional:
- torch.nn是torch.nnfunctional的一个封装，更好使用
one of basic building blocks : Containers
  - 六个模块：
  - Module: base class for all neural network modules
~~~
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module): #继承nn.Module
    def __init__(self): #初始化
        super().__init__() #一定要的，对父类也进行一个初始化
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 20, 5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        return F.relu(self.conv2(x))
        
model = Model()
x = torch.tensor(1.0)
output = model(x)
~~~

## 卷积层
<a id='卷积'></a>
- outchannel: 输出通道数就是卷积核的个数
