# ref: https://awsdocs-neuron.readthedocs-hosted.com/en/latest/containers/tutorials/build-run-neuron-container.html?highlight=dockerfile
build:
	docker build -f ${DOCKERFILE} -t ${IMAGE} .

test-container:
	docker run -p 80:8000 --device=/dev/neuron0 ${IMAGE} neuron-ls

run-container-interactive-single:
	docker run -it -p 80:8000 --device=/dev/neuron0 ${IMAGE} bash

run-container-interactive-multi:
	docker run -it -p 80:8000 --e "AWS_NEURON_VISIBLE_DEVICES=ALL" -e NEURON_RT_NUM_CORES=8 ${IMAGE} bash

run-container-interactive-multi-no-oci:
	docker run -it -p 80:8000 --device=/dev/neuron0 --device=/dev/neuron1 ${IMAGE} bash