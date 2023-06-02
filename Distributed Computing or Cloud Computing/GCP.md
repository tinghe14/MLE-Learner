[Python使用入门指南]（https://cloud.google.com/python/docs/getting-started?hl=zh-cn）

gcloud CLI:
- 是一组适用于 Google Cloud 的命令行工具。其中包含 gcloud、gsutil 和 bq，您可以使用它们通过命令行访问 Compute Engine、Cloud Storage、BigQuery 和其他产品和服务。这些工具既能以交互方式使用，也可以在自动化脚本中运行。


[GCP上的人工智能实用指南](https://github.com/tinghe14/apachecn-dl-zh/blob/master/docs/handson-ai-gcp/SUMMARY.md)

Course Content
- Google Cloud Platform的基础：
  - AI和GCP概述
  - 使用GCP组件的计算和处理
- 使用GCP的人工智能
  - XGBoost的机器学习应用
  - 使用Cloud AutoML
  - 构建大数据云机器学习引擎
  - 使用DialogFlow的智能对话应用
- Google Cloud Platform上的TensorFlow
  - 了解云TPU
  - 使用Cloud ML Engine实现TensorFlow模型
  - 构建预测应用
- 构建应用和即将发布的功能: 本节总结了从前几章获得的所有知识。 它由一章组成。 我们将使用 Google Cloud Platform（GCP）上的各种组件来构建端到端的 AI 应用。 本章提供了使用 GCP 快速构建生产就绪应用的一般过程。
  - 构建一个ai应用

# Google Cloud Platform的基础
### AI和GCP概述
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
