From: 小土堆 Pytorch

Course Content
- Intro-others
- 如何读取数据
- Dataset类
- TensorBoard使用
- Transforms
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
- package的法宝： dir(): 打开，看见里面有什么； help():这个函数的工具书
- os package: 做两个路径的连接
- self: 一个函数的变量不能传递给另外一个函数，self可以把这个指定的函数给后面的使用
~~~
import os 
root_dir = 'dataset/train'
label_dir = 'ants'
path = os.path.join(root_dir, label_dir) #dataset/train/ants
~~~

## 如何读取数据
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
