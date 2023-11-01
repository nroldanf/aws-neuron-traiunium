REGION := us-east-1
IMAGE_NAME := neuron-loka
REGISTRY = $(ACCOUNT).dkr.ecr.$(REGION).amazonaws.com
REPO_NAME = $(REGISTRY)/$(IMAGE_NAME)
IMAGE_TAGGED = $(REPO_NAME):latest

# ref: https://awsdocs-neuron.readthedocs-hosted.com/en/latest/containers/tutorials/build-run-neuron-container.html?highlight=dockerfile
# Get the images available
# aws ecr describe-images --region us-west-2 --registry-id 763104351884 --repository-name pytorch-training-neuron
ecr-login-trn:
	aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin 763104351884.dkr.ecr.us-west-2.amazonaws.com

build:
	docker build -f ${DOCKERFILE} -t ${IMAGE_NAME} .

test-container:
	docker run -p 80:8000 --device=/dev/neuron0 ${IMAGE_NAME} neuron-ls

run-container-interactive-single:
	docker run -it -p 80:8000 --device=/dev/neuron0 ${IMAGE_NAME} bash

run-container-interactive-multi:
	docker run -it \
		-p 80:8000 \
		-p 6006:6006 \
		-v /home/ubuntu/workdir/aws-neuron-traiunium/:/opt/app/traiunium \
		-v /home/ubuntu/workdir/aws-neuron-traiunium/.jupyter:/root/.jupyter \
		-e AWS_NEURON_VISIBLE_DEVICES=ALL \
		-e NEURON_RT_VISIBLE_CORES=0-1 \
		-e NEURON_RT_NUM_CORES=0 ${IMAGE_NAME} bash

run-container-interactive-multi-no-oci:
	docker run -it -p 80:8000 --device=/dev/neuron0 --device=/dev/neuron1 ${IMAGE_NAME} bash

run-jupyter:
	docker run -it \
		-v /home/ubuntu/workdir/aws-neuron-traiunium/:/opt/app/traiunium \
		-v /home/ubuntu/workdir/aws-neuron-traiunium/.jupyter:/root/.jupyter \
		-p 8888:8888 \
		-e "AWS_NEURON_VISIBLE_DEVICES=ALL" -e NEURON_RT_NUM_CORES=0 \
		${IMAGE_NAME} \
		jupyter notebook --allow-root --no-browser --port=8888