# MLE-Learner
I listed several technical components which I think is important for MLE role. I plan to tackle them one by one and organize my notes in this repository.

> No theory is perfect. Rather, it is a work in progress, always subject to further refinement and testing

markdown syntax shortcut:
- highlighter:
$`\textcolor{red}{\text{1}}`$ 
$`\textcolor{blue}{\text{2}}`$ 
$`\textcolor{green}{\text{3}}`$
- shortcut:
<a id='tag'></a> [pointer](#tag)

# Table of Contents (ongoing)
1. [coding language](#cl)
      - [python feature](#pf)
      - object orient programming
      - vectorize: numpy
      - [basic algorithm from scratch](#bafs1)
      - [pytorch syntax overview](#pso)
3. [software design and unit test](#sdut)
      - object orient design
      - git
      - unit test
      - web development
      - [computer architecture](#ca)
5. machine learning system design
      - educative course
      - youtuber 
      - technical blog
      - research paper
7. data manipulation
8. high preformance computing
9. [distributed computing or cloud computing](#dccc)
      - cluster
      - [GCP](#gcp)
      - docker
11. machine learning
      - basic algorithm from scrach
13. natural lanuage processing
      - basic algorithm from scrach
14. [deep learning](#dl)
      - basic algorithm from scratch
      - [deeplearning.ai course](#dlaic)
16. big data
17. AIML in production
      - educative course
      - deeplearning.ai course

# Coding Language
<a id='cl'></a>
## Python Feature
<a id='pf'></a>
## Basic Algorithms from scratch
<a id='bafs1'></a>
- [sorting]()
## pytorch syntax overview
<a id='pso'></a>
- my note - [Pytorch in Chinese](https://github.com/tinghe14/MLE-Learner/blob/cffde655d457b99a9a4a83f43e3b6f902bbaaf7a/Coding%20Language/Pytorch%20Syntax%20Overview/Pytorch.md): pycharm IDE + important modules for training and validating nn models including tensorboard

# Software Design and Unit Test
<a id='sdut'></a>
## Computer Architecture
<a id='ca'></a>
- [computer memory architecture](https://github.com/tinghe14/MLE-Learner/blob/7f686e86a8c86761d6068b0e69517632881cd9d7/Software%20Design%20and%20Unit%20Test/Computer%20Architecture/computer_memory_hierarchy.md)


# Distributed Computing or Cloud Computing
<a id='dccc'></a>
## GCP
<a id='gcp'></a>
- my note -[GCP in Chinese](https://github.com/tinghe14/apachecn-dl-zh/blob/master/docs/handson-ai-gcp/SUMMARY.md): not clear, recommend the second source listed here
- official GCP - [Github repo](https://github.com/GoogleCloudPlatform/cloudml-samples/tree/main): include information about project setup, docker image creation and sample code of pytorch
  - pytorch code sample using published container and built-in dataset, [tutorial](https://github.com/GoogleCloudPlatform/cloudml-samples/tree/main/pytorch/containers/published_container): container/docker image, can think as conda environment locally
  - pytorch code sample using custom container and built-in dataset,[tutorial](https://github.com/GoogleCloudPlatform/cloudml-samples/tree/main/pytorch/containers/custom_container): additional docker image creation file
  - pytorch code sample using custom container with hyperparameter tuning and built-in dataset,[tutorial](https://github.com/GoogleCloudPlatform/cloudml-samples/tree/main/pytorch/containers/hp_tuning): additional hyperparameter tuning
- tutorial - [Training PyTorch Transformers on Google Cloud AI Platform](https://nordcloud.com/tech-community/training-pytorch-transformers-on-google-cloud-ai-platform/): include best practices to packing code to GCP

# Deep Learning
<a id='dl'></a>
- deep learning specialization from deeplearning.ai 
  - my note - [Course #1 neural networks and deep learning]()
  - [improving deep neural networks: hyperparameter tuning, regularization and optmization]
  - [structing your machine learning project]
  - [natural language processing: building sequence models]

<!--
# Coding Language (foundation)
https://python-web-guide.readthedocs.io/zh/latest/base/basics.html
https://khuyentran1401.github.io/Efficient_Python_tricks_and_tools_for_data_scientists/Chapter6/logging_debugging.html
## Python Features
标题（描述下highlight）
1. educative module: Python 3: from beginner to advanced
      - [note]()
## Object Orient Programming
1. educative module: Learn Object-Oriented Programming in Python
      - [note]()
## Vectorize: Numpy
## Basic Algorithms from scratch

# Software Design and Unit Test （foundation）
## Object Orient Design
1. [educative module: grokking the low level design interview using OOD principles](https://www.educative.io/courses/grokking-the-low-level-design-interview-using-ood-principles)
  - 
## Git
https://zhuanlan.zhihu.com/p/34223150
## Unit Test
https://code.visualstudio.com/docs/python/testing
## Web Development
software framework: https://www.freecodecamp.org/news/what-is-a-framework-software-frameworks-definition/
建立一个网站hold自己的model


# Machine Learning System Design （high-priority）
## educative
## B站王树森
## 技术博客：卢明冬
## 各大公司博客
## 学术论文
page rank, learning to rank, arima time series modeling, ranking algorithm, 经典的recommendation system

# Data Manipulation
my notes are in Notion

# High Performance Computing 
# 看书

# Distributed Computing or Cloud Computing
## cluster
## GCP
## docker

# Machine Learning
## Basic Algorithms from scratch

# Natural Language Processing
## Basic Algorithms from scratch

# Big Data
## Spark

# AIML in Production
data science in productions: building scalable model pipelines
cousera: MLOOP

<!---
https://www.1point3acres.com/bbs/thread-997815-1-1.html
- 现在市场上有好多找做LLM背景人的坑。
我好奇这样背景的人和普通做NLP的人有什么主要的差异吗？
例如我这样的水货背景
- 3年前搞过一点NLP，会做常见的一些task（分类、问答、翻译什么的）。最近几年的进展都没怎么跟了。
- 明白古早版本的bert，transformer，gpt都是怎么工作的。
- 知道language model是怎么弄出来的（large的没碰过）
- 知道多机多卡的训练怎么写
- 会用一些已有推理框架onnx，tensorrt什么的捣鼓捣鼓模型上线
我可以大言不惭的说自己也是LLM背景的人吗？还是会被打回原型？
可能lz的能力能应付大多数工作了，但不足以在众多简历中被选出来，因为这些东西很多人都会。属实，感觉自己只能算个民科。研究方面完全没碰过。
很好的讨论，现在的公司精得很，感觉有没有百亿到千亿param 模型的实战的经验很容易就能在面试中看出来，在lz的基础上分享一些最近半年和相关资方打交道感受到的他们的期望和standard：
- 3年前搞过一点NLP，会做常见的一些task（分类、问答、翻译什么的）。最近几年的进展都没怎么跟了。
  --是否知道用10B以上LLM怎样便宜又有效的实现这些应用，LLM+RLHF/prompt engineering相比传统bert做基础任务有怎样的pros cons，怎样增强robustness/fairness
- 明白古早版本的bert，transformer，gpt都是怎么工作的
   --是否能在面试时不查api的情况下半小时pytorch/tf手撸朴素的bert/gpt实现 从 tokenizaiton/embedding/self attention and ffn 到beam search?
- 知道language model是怎么弄出来的（large的没碰过）
  --千亿规模模型训练都有哪些坑，数据清洗去重有哪些坑和调优技巧？怎么通过各种training dynamics的参数寻找适合的训练参数和训练早期发现不适合的模型参数？
- 知道多机多卡的训练怎么写.
   --megatron实现代码是否熟悉，知道如何修改？pipeline/tensor/data parallelism各项参数应该如何配置
- 会用一些已有推理框架onnx，tensorrt什么的捣鼓捣鼓模型上线
  --onnx/tensorrt/triton/pytorch2.0/deepspeed/fastertransformer用来部署百亿以上模型各有什么坑，如果需要4bit、8bit部署怎样为这些还不支持int8/int4实现相应的cuda kernel并调优超过cublas的水平？
可能他们进的早，我最近面openai和anthropic一类的公司 被问的比刚才列的还深
哎 确实有些面试造火箭的感觉 谁让现在这领域卷呢 不过倒也不用都精通，在一个方面比较专，其他方面能说出一些思考就行
我觉得偏工程的关心也没那么多
除了那几个Transformer的model外 (可以去Huggingface看) 也就是deepspeed zero了 ..... 我只会用data parallel 最多搞30-40B model 需要model/pipeline parallel 我也不知道哪个好
偏研究的东西就比较多了 最好还是经常看论文
比如比较新的positional encoding -> alibi / rotary 这种 会被考到
- 怎么说呢，比 LZ 水的搞 LLM 的人也有，比 LZ 强的面试进不去的也有。
- LZ 是想去搞 LLM，或者说是想去 OpenAI/Google Bard 这种吗？如果不是下面的建议不用看。
- 建议 LZ 跳出学生思维：不是我会这个技术，我就能去搞。
- 想明白这一点：你能为别人贡献什么，别人为什么需要你？
--->
