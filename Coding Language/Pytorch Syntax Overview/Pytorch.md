From: 小土堆 Pytorch

Course Content
- [Intro-others](#IO)
- [Dataset类](#dataset)
- [TensorBoard使用](#TB)
- [Transforms使用](#TF)
- [DataLoader使用](#DL)
- [nn.Module基本骨架](#nnM)
- [卷积层](#卷积)
- [池化层](#池化)
- [非线形激活](#非线性)
- [线性层及其他层](#线性)
- [sequential](#seq)
- [损失函数和反向传播](#损失)
- [优化器](#优化)
- [现有模型的使用](#现有)
- [网络模型的保存与读取](#保存)
- [完整模型训练套路](#完整)
- [利用GPU的训练](#GPU)
- [完整模型验证套路](#测试)
- 完结-看开源项目

## Intro-others
<a id='IO'></a>
- package的法宝： dir(库名): 查询库下的所有模块； help(函数名):这个函数的工具书
- os package: 做两个路径的连接
- self: 一个函数的变量不能传递给另外一个函数，self可以把这个指定的函数给后面的使用
- pycharm里查看函数具体信息的方式： help(函数名) 或者 常按control+鼠标移到这个函数的位置+点击蓝色链接进入__iniy__.py文件查看到该函数的定义
- 断点debug,执行到断点前
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
- ![卷积计算图](https://github.com/tinghe14/MLE-Learner/blob/598005d8334400207d29df4b00116a20c8c9f2cb/Coding%20Language/Pytorch%20Syntax%20Overview/%E5%8D%B7%E7%A7%AF%E8%AE%A1%E7%AE%97%E5%9B%BE.png)

## 池化层
<a id='池化'></a>
- MaxPool: 最大池化，下采样； MaxUnpool: 相反的，下采样
  - 保持主要特征，减少参数量
  - keneral size: 池化核，等于int时，生成一个正方形
  - dilation: 形成空洞卷积
  - stride: 步长，默认为kernal size
  - ceil modeL：没有满的时候，要不要做操作，默认是false

## 非线形激活
<a id='非线性'></a>
- 给网络引入非线性特征，因为非线性越多才可以画出更多的特征
- 常见：nn.ReLU, nn.Sigmoid

## 线性层及其他层
<a id='线性'></a>
- normalized layer: 归一化可以加快神经网络训练速度
  - nn.BatchNorm26
- recurrent layer
- transformer layer
- linear layer
- dropout layer
- embedding layer

## sequential
<a id='seq'></a>
- a sequential container

## 损失函数和反向传播
<a id='损失'></a>
- loss function: 计算实际输出和目标之间的差距；为我们更新输出提供一定的依据（反向传播）
  - nn.CrossEntropyLoss: 包含logsoftmax和NLLLoss，predict x是对每一类的得分，y是预测成哪一类
  - 计算反向传播是loss = nn.CrossEntropyLoss(x, y); loss.backward()

## 优化器
<a id='优化'></a>
'''
loss = nn.CrossEntropyLoss()
model = Model()
optim = torch.optim.SGD(model.parameters(), lr=0.01)
for epoch in range(20):
    running_loss = 0.0
    for data in dataloader:
        imgs, targets = data
        outputs = model(imgs)
        result_loss = loss(outputs, targets)
        optimizer.zero_grad() #把上一轮计算出来的grad清零
        # 不清零的话，梯度每次计算都会累积，会越来越大
        result_loss.backward()
        optimizer.step()
        running_loss = running_loss + result_loss
    print(running_loss)
'''

## 现有模型的使用
<a id='现有'></a>
- pretrained=False的预训练大模型，weight也不会为零，他们用的是一些固定的初始值比如xiaver initialization
'''
# 添加一层
vgg16_true = torchvision.models.vgg16(pretrained=True)
vfgg16_true.add_module('add_linear', nn.linear(1000, 10))
# 直接修改最后一层
vgg16_false.classifier[6] = nn.Linear(4096, 10)
'''

## 网络模型的保存与读取
<a id='保存'></a>
'''
vgg16 = torchvision.models.vgg16(pretrained=False)
# 保存方式1
# 保存了网络模型的结构也保存了网络模型的参数
torch.save(vgg16, 'vgg16_method1.pth')
# 加载模型1 需要在同一个地方能访问到模型是怎么训练的
model = torch.load('vgg16_model1.pth')

# 保存方式2 (官方推荐)
# 把网络模型的参数保存成字典
torch.save(vgg16.state_dict(), 'vgg16_method2.pth')
# 加载模型2
vgg16 = torchvision.models.vgg16(pretrained=False)
vgg16.load_state_dict(torch.load('vgg16_model2.pth'))
'''

## 完整模型训练套路
<a id='完整'></a>
'''
import torchvision
from torch.utils.tensorboard import SummaryWriter

# 准备数据集
from torch import nn
from torch.utils.data import DataLoader

train_data = torchvision.datasets.CIFAR10(root="../data", train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10(root="../data", train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)

# length 长度
train_data_size = len(train_data)
test_data_size = len(test_data)
# 如果train_data_size=10, 训练数据集的长度为：10
print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format(test_data_size))


# 利用 DataLoader 来加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 创建网络模型
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x
tudui = Tudui()

# 损失函数
loss_fn = nn.CrossEntropyLoss()

# 优化器
# learning_rate = 0.01
# 1e-2=1 x (10)^(-2) = 1 /100 = 0.01
learning_rate = 1e-2
optimizer = torch.optim.SGD(tudui.parameters(), lr=learning_rate)

# 设置训练网络的一些参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
epoch = 10

# 添加tensorboard
writer = SummaryWriter("../logs_train")

for i in range(epoch):
    print("-------第 {} 轮训练开始-------".format(i+1))

    # 训练步骤开始
    tudui.train()
    for data in train_dataloader:
        imgs, targets = data
        outputs = tudui(imgs)
        loss = loss_fn(outputs, targets)

        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            print("训练次数：{}, Loss: {}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试步骤开始
    tudui.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            outputs = tudui(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy

    print("整体测试集上的Loss: {}".format(total_test_loss))
    print("整体测试集上的正确率: {}".format(total_accuracy/test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)
    total_test_step = total_test_step + 1

    torch.save(tudui, "tudui_{}.pth".format(i))
    print("模型已保存")

writer.close()
'''

## 利用GPU的训练
<a id='GPU'></a>
- 训练方式一：
  - 找到网络模型
  - 数据（输入，标注）
  - 损失函数
- 调用.cuda()就行
- 训练方式二：
- .to(device)
- device = torch.device('cpu')
- device = torch.device('cuda')

## 完整模型验证套路
<a id='测试'></a>
- 利用已经训练好的模型，然后给他提供输入
'''
import torch
import torchvision
from PIL import Image
from torch import nn

image_path = "../imgs/airplane.png"
image = Image.open(image_path)
print(image)
image = image.convert('RGB')
transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                            torchvision.transforms.ToTensor()])

image = transform(image)
print(image.shape)

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x

model = torch.load("tudui_29_gpu.pth", map_location=torch.device('cpu'))
print(model)
image = torch.reshape(image, (1, 3, 32, 32))
model.eval()
with torch.no_grad():
    output = model(image)
print(output)

print(output.argmax(1))
'''
