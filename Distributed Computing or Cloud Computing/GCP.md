Crakking the Google Associate Cloud Engineer Certification: The course aims to help you understand the fundamentals of cloud computing

Course Content
- Introduction
- Creating Projects and IAM
- Billing Management
- Google Cloud CLI
- Compute Service Offerings
- Storage 
- Network Resources
- Cost Estimation
- Event Driven Services
- Monitoring and Logging 
- Miscellaneous Services

# Introduction
### Cloud Fundamentals: Service Categories
IAAS, Infrastructure As A Service:
- In IAAS, a cloud provider provides virtual access to its network resources to create our system. You will get access to storage, IPs, and firewalls, and the cloud provider manages all of these, in this case, Google.
PAAS, Platform As A Service:
- In this category of services, the Cloud provider will provide a well-defined and managed environment for your application. To simplify this, let’s take the example of Android Studio, Visual Studio, or Xcode. These all give you the components to create an app for their platform. You have to use them according to your idea of app development.
SAAS, Software As A Service:
- Any software or utility we use on the browser without installing natively on the system is referred to as “Software As A Service.” In the cloud computing domain, Google Cloud Platform does not provide any direct SAAS service; however, many Google Apps like Gmail, Maps, Calendar, and many more are hosted on Google Cloud, which are examples of SAAS. 
### Cloud Fundamentals: Virtualization
cloud computing is an environment that relies on various types of virtualization.
- server virtualization
  - The server is nothing but a remote computer with limited packages installed on it.
  - In server virtualization, logical separation of the central server is carried out, and each logical partition has its operating system, CPU, other supporting units, and on-demand GPUs.
- storage virtualization
  - virtual servers also need storage. Large physical hard drives are logically divided and attached to the servers in storage virtualization
  - Storage services are scalable, and in IAAS services, they are scaled according to the requirements without any extra service tickets to cloud providers.
- network virtualization
  - Cloud computing is about networking in the cloud, and server and storage virtualizations are needed to accomplish this primary task.
  - An organization needs several computers or servers to host its services in the cloud. So, all the computing power is provided per the load they receive on their software services. However, how do they identify each server, and how will one service communicate with the other? It will be using “networks” and “network identifiers.”
  - So, the ability to create different logical networks on the existing system is the foundation of cloud computing. Something called a “Virtual Private Cloud” is the root component of every cloud-based infrastructure. This is a logical separation, or a virtual network, under which all the servers, storage, and firewalls are organized, and the networking is carried out accordingly.
### Networking Fundamentals
- IP Address (Internet Protocol Address): Like postal addresses, servers also need an identity on the Internet. An IP address is a combination of unique numbers assigned to the server. For example, 8.8.8.8 is the public address of https://dns.google. Open the website and try finding the IPs of different websites.
- Ports: Ports are like the entry gates to the machine or server. Imagine a building where different entrances are created and dedicated to other people. Servers allow Internet or browser traffic using default ports 80 and 443. IP addresses and ports combined constitute the full location of the server. For example, 8.8.8.8:443 is the same as https://dns.google
- DNS (Domain Name System): “google.com” is a domain name, and 172.217.166.174 is the IP address of this domain. DNS is a database where all the mapping of IPs and domain names are maintained
- Firewall: The Firewall is a system through which we allow or deny entry and exit to and from our servers. We can use IPs, Ports, or a group of IPs to allow or deny traffic.
- VPC (Virtual Private Cloud): A VPC is a virtual network/space created using logical boundaries on top of the existing hardware. All the resources created using cloud computing are always in one VPC.
- Subnet: Subnet is short for “Sub Network.” A VPC is divided into small organizable units or sub-networks. There should be at least one subnetwork to create any resource in VPC. Sometimes, subnetworks are designed to represent a physical region or area logically.
- VM (Virtual Machine): This is the building block of any network infrastructure. A VM is a virtual computer connected to the VPC.
- Layers: A network consists of so many layers. The standard OSI model divides the network into seven layers. However, cloud computing abstracts the first three layers, and the rest of the layers are managed by us. In GCP, we mainly deal with Layer 4 and Layer 7.

# Creating Projects and IAM
### Creating Projects: Hands on
### IAM Roles and Permissions

# Compute Service Offerings

# Google Cloud Official Doc for Python
gcloud CLI:
- 是一组适用于 Google Cloud 的命令行工具。其中包含 gcloud、gsutil 和 bq，您可以使用它们通过命令行访问 Compute Engine、Cloud Storage、BigQuery 和其他产品和服务。这些工具既能以交互方式使用，也可以在自动化脚本中运行。
[Python使用入门指南]（https://cloud.google.com/python/docs/getting-started?hl=zh-cn）
