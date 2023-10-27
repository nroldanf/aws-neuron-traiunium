# Update OS packages 
sudo apt-get update -y

# Update OS headers 
sudo apt-get install linux-headers-$(uname -r) -y

# Install git 
sudo apt-get install git -y

# ****************************************************************
# NEURON CONFIGURATION
# ****************************************************************

# update Neuron Driver
sudo apt-get update aws-neuronx-dkms=2.* -y

# Update Neuron Runtime
# Runtime Library consists of the libnrt.so and header files
sudo apt-get install aws-neuronx-collectives=2.* -y
sudo apt-get install aws-neuronx-runtime-lib=2.* -y

# Update Neuron Tools
sudo apt-get install aws-neuronx-tools=2.* -y
# ****************************************************************

# Add PATH
export PATH=/opt/aws/neuron/bin:$PATH

# ONLY APPLICABLE FOR Trn1/Trn1n
# Install EFA Driver (only required for multi-instance training)
curl -O https://efa-installer.amazonaws.com/aws-efa-installer-latest.tar.gz 
wget https://efa-installer.amazonaws.com/aws-efa-installer.key && gpg --import aws-efa-installer.key 
cat aws-efa-installer.key | gpg --fingerprint 
wget https://efa-installer.amazonaws.com/aws-efa-installer-latest.tar.gz.sig && gpg --verify ./aws-efa-installer-latest.tar.gz.sig 
tar -xvf aws-efa-installer-latest.tar.gz 
cd aws-efa-installer && sudo bash efa_installer.sh --yes 
cd 
sudo rm -rf aws-efa-installer-latest.tar.gz aws-efa-installer

# OCI runtime
sudo apt install -y golang && \
    export GOPATH=$HOME/go && \
    go get github.com/joeshaw/json-lossless && \
    cd /tmp/ && \
    git clone https://github.com/awslabs/oci-add-hooks && \
    cd /tmp/oci-add-hooks && \
    make build && \
    sudo cp /tmp/oci-add-hooks/oci-add-hooks /usr/local/bin/

# install OCI hook software

# INF1
# sudo apt-get install aws-neuron-runtime-base -y

# TRN1
sudo apt-get install aws-neuronx-oci-hook -y

# For docker runtime
sudo cp /opt/aws/neuron/share/docker-daemon.json /etc/docker/daemon.json
sudo service docker restart