[Python使用入门指南]（https://cloud.google.com/python/docs/getting-started?hl=zh-cn）

gcloud CLI:
- 是一组适用于 Google Cloud 的命令行工具。其中包含 gcloud、gsutil 和 bq，您可以使用它们通过命令行访问 Compute Engine、Cloud Storage、BigQuery 和其他产品和服务。这些工具既能以交互方式使用，也可以在自动化脚本中运行。


[GCP上的人工智能实用指南](https://github.com/tinghe14/apachecn-dl-zh/blob/master/docs/handson-ai-gcp/SUMMARY.md)

Course Content
- Google Cloud Platform的基础：
  - [AI和GCP概述](#概述)
  - [使用GCP组件的计算和处理](#计算和处理)
- 使用GCP的人工智能
  - [XGBoost的机器学习应用](#XGBoost)
  - 使用Cloud AutoML
  - [构建大数据云机器学习引擎](#大数据)
  - 使用DialogFlow的智能对话应用
- Google Cloud Platform上的TensorFlow
  - 了解云TPU
  - 使用Cloud ML Engine实现TensorFlow模型
  - 构建预测应用
- 构建应用和即将发布的功能: 本节总结了从前几章获得的所有知识。 它由一章组成。 我们将使用 Google Cloud Platform（GCP）上的各种组件来构建端到端的 AI 应用。 本章提供了使用 GCP 快速构建生产就绪应用的一般过程。
  - 构建一个ai应用

# Google Cloud Platform的基础
### AI和GCP概述
<a id='概述'></a>
cloud first优先策略
- 优势： 最低的前期成本（由于服务全天候可用，因此存储和计算基础架构几乎不受限制），弹性容量（灵活的数据容量管理），全局链接（只要可以使用互联网连接以及适当的认证/授权，就可以在全球虚拟访问作为云上可用的基础结构），无缝升级，无服务器DevOps(他们可以根据数据量和计算需求，根据应用范围简单地规定存储和计算需求。这些团队无需担心任何部署，他们将以最少的配置和扩展时间开始使用服务)，快速发布时间（有了前面列出的所有优点，采用cloud first策略将各种概念和原型的发布时间降至最低）

gcp概述
- 尽管任何云平台都可以作为虚拟服务通过网络使用，但其核心是系统和物理资源
- GCP 上的所有资源（例如存储和计算）都被视为服务。 该范例称为一切即服务（XaaS）。 这包括 IaaS，平台即服务（PaaS）等
- GCP提供了web界面控制台，命令行界面（cli）和cloud shell,与其各种服务进行交互
- sdk(云服务开发套件)提供了一个称为gcloud的cli工具，可用于执行所有配置并与平台进行交互。他也可以用于开发工作流程的管理。cloud shell提供了与gcp进行交互的类似界面，cloud shell是一个基于浏览器的临时hshell环境，可以从云控制台内部进行访问

数据
- 可以与人类能力相匹配的智能机器的一般概念是在大约一个世纪前提出的。但是，研究受到的可用数据存储和处理能力的限制，因为构成ai基础的机器学习模型需要大量的数据来进行训练，并且需要大量的处理能力来进行算法计算。数据是ai构建模块的核心和焦点。数据容量分为三个区域：存储，处理和数据驱动的操作
- 存储：由于采用了云计算，通用存储容量的可访问性也得到了显著提高。在云计算范例中，存储作为服务可用，不需要采购和管理与存储相关的基础架构
- 处理：由于分布式计算框架，我们还看到了整体处理能力的提高。处理单元分布在各种机器上，以进行并行处理和计算。框架负责跟踪跨节点的计算，并整合从可行见解中的出的结果

### 使用GCP组件的计算和处理
<a id='计算和处理'></a>
了解计算选项
- GCP 提供了各种计算选项来部署您的应用，并使用实际的 Google 基础架构来运行您的应用。 可用的选项如下： 基础架构即服务（laaS）,容器，平台即服务（PaaS）
- 所有计算选项均与其他gcp服务和产品进行通信，例如存储，网络，stackdriver,安全性和大数据产品套件。根据给定应用的需求，从compute engine, kubernetes engine, app engine和cloud functions中选择适当的计算选项。
- google计算选项可以帮助你在google基础架构上运行多种大小的虚拟机并对其进行自定义。它使你能够运行容器化的应用，并且如果你不必照顾与基础架构相关的项目，则可以直接在引擎上部署代码。
- 接下来我们将详细介绍以下计算选项：
  - 计算引擎compute engine
  - 应用引擎
  - cloud functions
  - kubernetes引擎

计算引擎compute engine
- laaS. compute engine是在 google基础架构中运行的虚拟机
- google cloud提供的所有区域都可以使用compute engine.它具有永久性磁盘和本地固态驱动器ssd的存储选项。
  - ssd内部内置芯片上集成电路，不包含然和旋转头或磁盘驱动器。与硬盘驱动器相比，ssd更耐用，读取时间更快
  - 永久磁盘是一种网络存储，最多可以扩展到64tb，而本地ssd是加密驱动器，它实际上已经连接到服务器，并且可以扩展到3tb
- 用户可以使用 Linux 或 Windows 操作系统启动 Compute Engine。 这些实例可以使用 CPU，GPU 和 TPU 启动，并且由于基础结构是由 Google 提供的，因此用户可以进行操作系统级的自定义。
- 用户可以在 Compute Engine 中创建托管和非托管实例组：
  - 受管实例组将始终包含相同的虚拟机，并支持自动扩展，高可用性，滚动更新等。
  - 非托管实例组可以包含具有不同配置的计算机。 用户可以在创建托管实例组时使用实例模板，但不能与非托管实例组一起使用。
  - 建议选择一个受管且统一的实例组，直到在同一池中非常需要不同配置的计算机为止。

Compute Engine 和 AI 应用
- 在为 AI（ML）应用训练模型时，始终需要功能强大的机器，以通过提供充足的训练数据并减少训练模型的时间来提高模型的效率。
- app engine是 Google Cloud 提供的 PaaS； 它是一个完全托管的无服务器应用平台。
  - 您可以将 App Engine 视为可用于部署的基础架构； 开发人员只需专注于构建应用并将其部署到 App Engine 上，其他所有事情都将得到解决。 App Engine 具有出色的功能，例如自动缩放，流量拆分，应用安全，监视和调试-所有这些功能对于部署，保护和扩展任何应用都是必不可少的。 使用 Cloud SDK 和 IntelliJ IDEA 之类的工具，开发人员可以直接连接到 App Engine 并执行诸如调试源代码和运行 API 后端之类的操作。 App Engine 的限制之一是无法自定义其操作系统
计算引擎compute engined饿
- App Engine 标准环境应用在沙盒环境中运行，并支持运行 Python，Java，Node.js，Go 和 PHP 应用。 另一方面，App Engine 灵活环境应用在 Google Compute Engine 虚拟机上的 Docker 容器中运行，除了标准环境支持的语言外，还支持运行 Ruby 和 .NET 应用。
- App Engine 对于部署任何 Web 或移动应用非常有用。 根据资源的使用情况，基础架构会自动扩展，Google 只会针对已使用的应用收费

App Engine 和 AI 应用
-  App Engine 上运行任何移动或 Web 应用时，在许多用例中，这些应用都需要 AI。 在 App Engine 中部署应用时可以实现这些目标。 该服务可以与云终结点一起部署，$`\textcolor{red}{\text{而 Python 应用可以在 App Engine 中部署，从而加载训练有素的机器学习模型}}`$ 。 通过 App Engine 访问模型后，该服务可以将请求发送到 Python 应用并以一致的方式获取响应。

Cloud Functions 和 AI 应用
- 在运行任何应用时，如果用户希望基于特定事件调用 Cloud ML 或 Cloud Vision 的 API，则可以使用 Cloud Functions。

Kubernetes Engine
- Kubernetes Engine 是 Google Cloud 提供的一项托管服务； 它用于部署和运行容器化的应用。 以下是 Kubernetes Engine 的功能：
  - 在 Kubernetes 集群下，Google 实际上正在运行 Compute Engine，因此我们在 Compute Engine 上拥有的大多数优势将与 Kubernetes Engine 一起使用，并提供其提供的其他服务。
  - 在 Kubernetes 集群中，可以使用具有自定义 OS 映像的虚拟机，并且集群将自动缩放自定义映像。
  - Kubernetes 集群具有高度的安全性，并获得了 HIPAA 和 PCI DSS 3.1 的支持。
  - 它支持常见的 Docker 映像和私有容器注册表，用户可以通过该注册表访问私有 Docker 映像。
  - Kubernetes 集群可以与 Stackdriver 集成在一起，以实现对集群的监视和日志记录。

Kubernetes Engine 和 AI 应用
- 在为 Al（ML）应用训练模型时，始终需要功能强大的机器，以通过提供充足的训练数据并减少训练模型的时间来提高模型的效率。 $`\textcolor{red}{\text{可以使用GPU构建Kubernetes集群，以训练模型并运行ML工作负载}}`$。 这可以使许多机器学习应用受益，这些应用需要具有强大 GPU 机器的托管容器化集群。
  
进入存储选项
- GCP 提供了各种存储选项来存储您的应用数据。 不同的应用具有不同的存储需求，并且取决于应用，性能得以提高。
- 为您的应用选择正确的存储选项很重要。 根据 Google 中可用的存储选项，以下图表将帮助您确定正确的存储选项：
![存储选项](https://github.com/tinghe14/MLE-Learner/blob/a8a5e5599aaa3e7115803642f4d28b5b4c129f64/Distributed%20Computing%20or%20Cloud%20Computing/GCP%E5%AD%98%E5%82%A8%E9%80%89%E9%A1%B9.png)

cloud storage
- 云存储是 GCP 提供的对象存储。 以下是云存储的功能：
  - 它可以存储任何数量的数据和各种格式的数据，包括结构化数据，非结构化数据，视频文件，图像，二进制数据等。
  - 用户可以根据以下要求将数据存储在 Cloud Storage 中的四个不同的存储桶中，即多区域存储，区域存储，近线存储和冷线存储。如果数据在世界范围内经常访问，则转到“多区域”存储桶; 如果经常在同一地理区域访问数据，则进入“区域”存储桶。 对于每月访问一次的数据，请使用 Nearline，对于每年访问一次的数据，请使用 Coldline 存储桶; 选择桶很重要，因为与之相关的成本
  - Cloud Storage 提供了 API 和工具，用于进出数据传输。
  - 用户可以使用gsutil工具从本地传输数据，也可以使用云服务从其他云传输数据。
  - BigQuery 和 Dataproc 等服务可以访问 Cloud Storage 中存储的数据，以创建表并将其用于处理中
  - 凭借其所有功能，云存储是 GCP 上最常用的存储选项，也是最便宜的存储选项之一。 根据存储类别和访问模式，其价格从每月每 GB 0.007 美元到每月每 GB 0.036 美元不等

Cloud Storage 和 AI 应用
- 云存储可以在各种 AI 和 ML 用例中提供帮助。 大多数大数据迁移或现代数据平台都使用 Cloud Bigtable 构建其 NoSQL 数据库。 例如，Spark ML 应用将访问 Cloud Bigtable 中的数据并将结果存储在其中。 云存储已经用于基因组学，视频转码，数据分析和计算等用例。
- Cloud Bigtable 是 GCP 提供的完全托管的 NoSQL 数据库系统。 它可以以极低的延迟和高吞吐量扩展到 PB 级的数据

BigQuery 和 AI 应用
- BigQuery ML 是 BigQuery 机器学习的一种形式，它具有一些内置算法，可以直接在 SQL 查询中用于训练模型和预测输出。 BigQuery ML 当前支持分类模型的线性回归，二进制逻辑回归和多类逻辑回归

Cloud Dataproc
- Cloud Dataproc 是一个完全托管的 Hadoop 和 Spark 集群，可以在几秒钟内旋转。 Cloud Dataproc 是一个自动扩展集群，可用于非常有效地运行 Hadoop，Spark 以及 AI 和 ML 应用。 在高峰时段，可以根据使用情况将节点添加到群集，并且在需求较低时可以进行缩减。
- Dataproc 与其他服务集成，例如云存储，BigQuery，Stackdriver，身份和访问管理以及网络。 这使得群集的使用非常容易且安全

其他的存储方式暂时我不会设计，有需要继续阅读[这个章节](https://github.com/tinghe14/apachecn-dl-zh/blob/master/docs/handson-ai-gcp/02.md)

建立 ML 管道
- 让我们来看一个详细的示例，在该示例中，我们将建立一条端到端的管道，从将数据加载到 Cloud Storage，在其上创建 BigQuery 数据集，使用 BigQuery ML 训练模型并对其进行测试。 在此用例中，我们将使用逻辑回归模型来查找潜在客户转化概率。 您可以使用选择的任何合适的数据集并遵循此示例。
- 数据先加载到 Cloud Storage 和 BigQuery 中以及对模型进行训练并使用潜在客户数据进行测试的端到端过程
![将数据加载到cloud storage](https://github.com/tinghe14/MLE-Learner/blob/676b90e230f2ab09828334cb2b426f6b5eb70e63/Distributed%20Computing%20or%20Cloud%20Computing/%E5%B0%86%E6%95%B0%E6%8D%AE%E5%8A%A0%E8%BD%BD%E5%88%B0%20Cloud%20Storage.png)
![将数据加载到big query 1](https://github.com/tinghe14/MLE-Learner/blob/676b90e230f2ab09828334cb2b426f6b5eb70e63/Distributed%20Computing%20or%20Cloud%20Computing/%E5%B0%86%E6%95%B0%E6%8D%AE%E5%8A%A0%E8%BD%BD%E5%88%B0%20BigQuery%201.png)
![将数据加载到big query 2](https://github.com/tinghe14/MLE-Learner/blob/676b90e230f2ab09828334cb2b426f6b5eb70e63/Distributed%20Computing%20or%20Cloud%20Computing/%E5%B0%86%E6%95%B0%E6%8D%AE%E5%8A%A0%E8%BD%BD%E5%88%B0%20BigQuery%202.png)
![将数据加载到big query 3](https://github.com/tinghe14/MLE-Learner/blob/676b90e230f2ab09828334cb2b426f6b5eb70e63/Distributed%20Computing%20or%20Cloud%20Computing/%E5%B0%86%E6%95%B0%E6%8D%AE%E5%8A%A0%E8%BD%BD%E5%88%B0%20BigQuery%203.png)
![训练模型 1](https://github.com/tinghe14/MLE-Learner/blob/676b90e230f2ab09828334cb2b426f6b5eb70e63/Distributed%20Computing%20or%20Cloud%20Computing/%E8%AE%AD%E7%BB%83%E6%A8%A1%E5%9E%8B%201.png)
![训练模型 2](https://github.com/tinghe14/MLE-Learner/blob/676b90e230f2ab09828334cb2b426f6b5eb70e63/Distributed%20Computing%20or%20Cloud%20Computing/%E8%AE%AD%E7%BB%83%E6%A8%A1%E5%9E%8B%202.png)
![评估模型](https://github.com/tinghe14/MLE-Learner/blob/676b90e230f2ab09828334cb2b426f6b5eb70e63/Distributed%20Computing%20or%20Cloud%20Computing/%E8%AF%84%E4%BC%B0%E6%A8%A1%E5%9E%8B.png)
![测试模型](https://github.com/tinghe14/MLE-Learner/blob/676b90e230f2ab09828334cb2b426f6b5eb70e63/Distributed%20Computing%20or%20Cloud%20Computing/%E6%B5%8B%E8%AF%95%E6%A8%A1%E5%9E%8B.png)

# 使用GCP的人工智能
### XGBoost的机器学习应用
<a id='XGBoost'></a>
一种是建立并使用有效的统计模型来表示涵盖最重要场景的所有复杂数据依赖关系，另一种是可伸缩地构建那些模型以确保它们在越来越大的数据集上工作。 机器学习的可伸缩性使您能够通过并行和分布式计算进行快速学习，并且还提供了有效的内存利用率。结果是一个单一模型，提供了多个模型的附加输出。 集成模型（也称为基础学习器）可以使用相同的学习算法，也可以使用不同的算法。 在本章中，您将学习如何使用 Python 实现 XGBoost 算法。 您还将学习有关梯度提升的概念以及它们如何支持 XGBoost 算法的知识。
本章将涵盖以下主题：

XGBoost 库概述
- 训练和存储 XGBoost 机器学习模型
- 用已训练的 XGBoost 模型
- 使用 XGBoost 库构建推荐系统

集成学习
- 集成学习算法结合了多个基本模型以产生最佳的预测模型。 与尝试从训练数据中学习假设的普通机器学习方法相反，集成方法尝试构建和组合许多假设。 使用集成方法时，重点放在泛化上，由于算法的特定性质或由于选择的训练集而导致的偏差方差折衷，有时基础学习器会忽略它。
- bagging和boosting是广泛使用的集成方法。 这些是一种简单的组装技术，使用一些模型平均技术来共同构建许多独立的预测变量/模型/学习器（例如平均加权，多数或平均法线）
- 另一个更大的问题是如何确保基础学习器不相互关联。 确保这些基本共同学习器的预测彼此独立是很重要的。 否则，它将排除整体建模的优化预测模型。??

集成学习如何决定最佳预测模型？
- 产生的误差尽可能低时，确定最佳模型，而损失函数的较低值确定最小值。 损失函数用于衡量预测模型可以预测预期结果的程度。 查找最小函数点的最常见方法是梯度下降。总而言之，我们必须首先了解导致模型误差的原因，才能真正了解集成模式背后的原因。 任何模型的误差都可以在数学上分为三种类型。
- 可减少的误差 – 偏差： 偏差是预测值和实际值之间的距离。 如果模型具有很高的偏差度，则意味着该模型过于简单，无法捕获支撑数据的复杂性。
- 可减少的误差 – 方差： 当模型在经过训练的数据集上良好但在新数据集（例如测试数据集或验证数据集）上表现不佳时，就会发生方差。 方差告诉我们如何分散实际值。 高方差会导致过拟合，这意味着算法模型中训练数据中存在随机噪声。 如果模型显示出很高的方差，则它会变得非常灵活，并适应训练集的数据点。？？
- 不可减少的误差： 不可减少的误差是无论您使用哪种机器学习算法都无法将其最小化的误差。 它们通常是由可能影响输出变量的未知变量引起的。 改善不可减少误差的预测的唯一方法是识别和预测那些外部影响。
![总误差](https://github.com/tinghe14/MLE-Learner/blob/693d5029f101d72c798425dfa37e098d0741e6af/Distributed%20Computing%20or%20Cloud%20Computing/%E6%80%BB%E8%AF%AF%E5%B7%AE.png)

XGBoost
- 以下是 XGboost 使其独特的一些重要功能：
  - 并行化：在 XGBoost 中并行构建顺序树。 在 XGBoost 中，为了缩短运行时间，通过初始化全局扫描并使用所有实例的并行线程进行排序来交换循环的顺序。 此开关通过抵消任何并行开销来提高算法效率??
  - 停止标准：在梯度提升框架中，用于树分割的停止标准基于分割时的负损失函数。 但是，对于 XGBoost，它将按指定的max_depth参数开始向后修剪树。 这是一种深度优先的方法，可以提高算法的整体性能和效率??
  - 最大硬件资源使用率：XGBoost 旨在最大程度地利用硬件资源。 它通过每棵树中的内部缓冲区利用系统缓存来存储梯度统计信息。 其他改进包括计算核外和优化可用磁盘空间，同时处理不适合内存的大数据帧??

训练和存储 XGBoost 机器学习模型
- 在本节中，我们将研究如何使用 Google AI Hub 训练和存储机器学习模型。 AI Hub 是一站式存储，用于检测，共享和部署机器学习模型。 它是可重用的模型目录，可以快速安装在 AI 平台执行环境中。 该目录包含基于以下通用框架的模型设计的汇编：TensorFlow，PyTorch，Keras，scikit-learn 和 XGBoost。 每种模型都可以包装由 GPU 或 TPU，Jupyter 笔记本和 Google 自己的 AI API 支持的深度学习 VM，格式可以在 Kubeflow 中实现。借助.../text_classification_rapids_framework.py处的代码，我们将 XGBoost 与 RAPIDS 框架一起用于文本分类。
- 从算法上讲，此代码执行以下步骤：
  - 导入了必要的包。 在较高级别上，此代码使用OS，google.cloud，cudf(RAPID)，sklearn，pandas和xgboost。 它还会导入pynvml，这是一个 Python 库，用于低级 CUDA 库，用于 GPU 管理和监视。
  - 接下来，代码将安装miniconda库和 RAPIDS 平台，然后设置 NVIDIA GPU 所需的一些环境变量。
  - 代码的下一部分将设置访问 Google API 所需的一些常量，例如project id和bucket id。
  - 然后，该代码从 GCS 存储桶中下载训练数据（text_classification_emp.csv）。 然后将其存储在本地作业目录中以供进一步使用。
  - 代码的下一部分使用\n分割 CSV，并创建两个数组，一个数组用于标签（目标变量），另一个数组用于文本（预测变量）。
  - 然后，它创建一个 Pandas DataFrame，然后将其转换为与基础 GPU 兼容的 CUDF DataFrame。 这样做是为了确保所有其他操作都利用基础 GPU。
  - 然后，按照 80%-20% 的规则将数据分为训练和测试数据集。
  - 然后，标签编码器用于将标签编码为向量值。
  - 之后，在字符级别上计算 TF-IDF。
  - 最后，使用 XGBoost 库训练模型。

要提交前面的代码来训练模型，您必须运行以下命令。 此命令是标准google-ai-platform CLI，该 CLI 提交训练说明以在 Google Cloud AI 平台上训练任何模型：
~~~
gcloud ai-platform jobs submit training $JOB_NAME \
--job-dir=$JOB_DIR \
--package-path=$TRAINING_PACKAGE_PATH \
--module-name=$MAIN_TRAINER_MODULE \
--region=$REGION \
--runtime-version=$RUNTIME_VERSION \
--python-version=$PYTHON_VERSION \
--config=config.yaml
~~~

可以按照以下方法将那边的环境变量设置为job.properties，并且在运行gcloud ai-platform作业之前必须先获取job.properties的源。 可以在以下代码中看到一个示例：
~~~
PROJECT_ID=test-project-id
BUCKET_ID=ml-assets
JOB_NAME=gpu_text_classification_$(date +"%Y%m%d_%H%M%S")
JOB_DIR=gs://${BUCKET_ID}/xgboost_job_dir
TRAINING_PACKAGE_PATH="text_classification"
MAIN_TRAINER_MODULE=text_classification.train
REGION=us-central1
RUNTIME_VERSION=1.13
PYTHON_VERSION=3.5
~~~

特定于 GPU 的config.yml文件的内容如下：
~~~
trainingInput:
  scaleTier: CUSTOM
  # Configure a master worker with 4 K80 GPUs
  masterType: complex_model_m_gpu
  # Configure 9 workers, each with 4 K80 GPUs
  workerCount: 9
  workerType: complex_model_m_gpu
  # Configure 3 parameter servers with no GPUs
  parameterServerCount: 3
  parameterServerType: large_model
~~~

包的结构如下块所示：
~~~
text_classification
|
|__ __init__.py
|__ config.yml
|__ run.sh
|__ job.properties
|__ train.py
~~~

使用已训练的 XGBoost 模型
![使用XGBoost1](https://github.com/tinghe14/MLE-Learner/blob/9427015c89ded9dbfe1bd9f4d29c814a68d433a1/Distributed%20Computing%20or%20Cloud%20Computing/xgboost%E4%BD%BF%E7%94%A81.png)
![使用XGBoost2](https://github.com/tinghe14/MLE-Learner/blob/9427015c89ded9dbfe1bd9f4d29c814a68d433a1/Distributed%20Computing%20or%20Cloud%20Computing/xgboost%E4%BD%BF%E7%94%A82.png)
![使用XGBoost3](https://github.com/tinghe14/MLE-Learner/blob/9427015c89ded9dbfe1bd9f4d29c814a68d433a1/Distributed%20Computing%20or%20Cloud%20Computing/xgboost%E4%BD%BF%E7%94%A83.png)
![使用XGBoost4](https://github.com/tinghe14/MLE-Learner/blob/9427015c89ded9dbfe1bd9f4d29c814a68d433a1/Distributed%20Computing%20or%20Cloud%20Computing/xgboost%E4%BD%BF%E7%94%A84.png)

### 构建大数据云机器学习引擎
<a id='大数据'></a>
