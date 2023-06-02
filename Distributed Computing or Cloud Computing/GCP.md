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
