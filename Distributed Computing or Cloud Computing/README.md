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
    - [Youtube, overview of documentation](https://www.youtube.com/watch?v=VRQXIiNLdAk): use pretrained docker image, create a docker file, and share the skeleton of image file; previous episode contains how to upload local data to cloud storage
        - tutorial - [steps by steps](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqbkRaWElYeVhSdnNuZS1uUENYZHlqQ1VHeGJtZ3xBQ3Jtc0tremd0WGw0X1VxcTlYMzhfQzg4bkpxaDZfN25jSFZHUDg0bDZyUzRKdHB6bk5VbzYzRkJkQ0RGR1FqcHFOanMzVWdJdS05bF9UNlFsUUlCR0lRTVpqd3ZZQUpvUjRyZnRIV2tmaHVNM3hwNWt1UzFUVQ&q=https%3A%2F%2Fgoo.gle%2F3w7kGvV&v=VRQXIiNLdAk)
    - tutorial - [setup](https://cloud.google.com/vertex-ai/docs/start/cloud-environment)
    - tutorial - [export environment at GCI](https://cloud.google.com/vertex-ai/docs/tutorials/text-classification-automl)
    - tutorial - [Create a Python training application for a prebuilt container](https://cloud.google.com/vertex-ai/docs/training/create-python-pre-built-container?_ga=2.78161364.1169932329.1685983386-1703019298.1684697406&_gac=1.123091193.1685635930.Cj0KCQjw4NujBhC5ARIsAF4Iv6dUxaJlcW7I1ourBHTusyrNz8FSF2JwF3IkOTpaH20BRe0oxoX7LkUaAremEALw_wcB)
        - tutorial - [Containerize and run training code locally](https://cloud.google.com/vertex-ai/docs/training/containerize-run-code-local)
        - tutorial - [Push the container to Artifact Registry](https://cloud.google.com/vertex-ai/docs/training/create-custom-container#build-and-push-container)
