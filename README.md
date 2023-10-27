# Neuron SDK

## Neuron Compiler

The Neuron Compiler optimizes ML models to run on Neuron devices. It accepts Machine Learning models in various formats (TensorFlow, MXNet, PyTorch, XLA HLO). The Neuron compiler is invoked within the ML framework, where ML models are sent to the compiler by the `Neuron Framework plugin`. The resulting compiler artifact is called a `NEFF file (Neuron Executable File Format)` that in turn is loaded by the `Neuron runtime` to the Neuron device.

## Neuron Runtime

Neuron Runtime is responsible for executing ML models on Neuron Devices. Neuron Runtime determines which NeuronCore will execute which model and how to execute it. Configuration of the Neuron Runtime is controlled through the use of Environment variables at the process level. Neuron runtime consists of kernel driver and C/C++ libraries which provides APIs to access Inferentia and Trainium Neuron devices. 

The Neuron ML frameworks plugins for TensorFlow, PyTorch and Apache MXNet use the Neuron runtime to load and run models on the `NeuronCores`. Neuron runtime loads `compiled` deep learning models, also referred to as `Neuron Executable File Format (NEFF)` to the Neuron devices and is optimized for `high-throughput and low-latency`.

Neuron Runtime Library consists of the libnrt.so and header files. These artifacts are version controlled and installed via the aws-neuronx-runtime-lib package. After installing the package, the binary (libnrt.so) is found in /opt/aws/neuron/lib

Ref:
- [Runtime Config Options](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-runtime/nrt-configurable-parameters.html#nrt-configuration)
- [Runtime Developer Guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-runtime/nrt-api-guide.html)

### Expose all neuron devices to the container

Neuron devices are exposed to the containers using the –device option in the docker run command. Docker runtime (runc) does not yet support the ALL option to expose all neuron devices to the container. In order to do that an environment variable, `AWS_NEURON_VISIBLE_DEVICES=ALL` can be used.

**Context:**
- The hooks enable Containers to be aware of events in their management lifecycle and run code implemented in a handler when the corresponding lifecycle hook is executed. There are two hooks that are exposed to Containers:
    - `prestart`
    - `PostStart`
    - `PreStop`

**Downsides**
- Multiple container applications running in the same host can share the devices but the cores cannot be shared. This is similar to running multiple applications in the host.
- In the kubernetes environment the devices cannot be shared by multiple containers in the pod.

Ref:
- [OCI Hook](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/containers/tutorials/tutorial-oci-hook.html#tutorial-oci-hook)
- [Open Container Iniciate - OCI](https://github.com/opencontainers/runtime-spec)
- [Container Lifecycle Hooks](https://kubernetes.io/docs/concepts/containers/container-lifecycle-hooks/)
- [Container Runtime](https://kubernetes.io/docs/concepts/containers/#container-runtimes)


## Neuron Collectives

Neuron Collectives refers to a set of libraries used to support collective compute operations within the Neuron SDK.


## Frameworks that support Neuron

### Huggingface

https://huggingface.co/blog/pytorch-xla