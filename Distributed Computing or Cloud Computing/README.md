# Material that I have learnt
1. [GCP in Chinese](https://github.com/tinghe14/apachecn-dl-zh/blob/master/docs/handson-ai-gcp/SUMMARY.md): not clear, waste of time
2. official GCP repo: clear but the docker image part has been outdated
    - [Github repo](https://github.com/GoogleCloudPlatform/cloudml-samples/tree/main): include information about project setup, docker image creation and sample code of pytorch,BUT container part is too old
    - pytorch code sample using published container and built-in dataset, [tutorial](https://github.com/GoogleCloudPlatform/cloudml-samples/tree/main/pytorch/containers/published_container): container/docker image, can think as conda environment locally
    - pytorch code sample using custom container and built-in dataset,[tutorial](https://github.com/GoogleCloudPlatform/cloudml-samples/tree/main/pytorch/containers/custom_container): additional docker image creation file
    - pytorch code sample using custom container with hyperparameter tuning and built-in dataset,[tutorial](https://github.com/GoogleCloudPlatform/cloudml-samples/tree/main/pytorch/containers/hp_tuning): additional hyperparameter tuning
3. official GCP tutorial: updated but It can't follow
    - [container part](https://cloud.google.com/ai-platform/docs/getting-started-keras)
    - [Training PyTorch Transformers on Google Cloud AI Platform](https://nordcloud.com/tech-community/training-pytorch-transformers-on-google-cloud-ai-platform/): include [best practices](https://cloud.google.com/ai-platform/training/docs/packaging-trainer) to packing code to GCP, and [recommended project structure](https://cloud.google.com/ai-platform/training/docs/packaging-trainer#project-structure)
4. official Vertex AI tutorial: most updated and clear to follow
    - [Youtube, overview of documentation](https://www.youtube.com/watch?v=VRQXIiNLdAk): use pretrained docker image and share the skeleton of image file
    - tutorial - [setup](https://cloud.google.com/vertex-ai/docs/start/cloud-environment)
    - tutorial - [export environment at GCI](https://cloud.google.com/vertex-ai/docs/tutorials/text-classification-automl)
    - right now at uploading file, which they don't accept tsv, need to convert to csv and change corresponding code first , https://www.geeksforgeeks.org/python-convert-tsv-to-csv-file/
