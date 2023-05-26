From: 小土堆 Pytorch

Course Content
- [Intro-others](#IO)
- [Dataset类](#dataset)
- [TensorBoard使用](#TB)
- [Transforms使用](#TF)
- DataLoader水涌
- nn.Module基本骨架
- 卷积层
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
- pycharm里查看函数具体信息的方式： help(函数名) 或者 鼠标移到这个函数的位置+常按control+点击蓝色链接进入官方doc
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

