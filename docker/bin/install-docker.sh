#!/bin/bash
sudo apt-get remove -y "*docker*" 
sudo apt-get remove -y containerd
sudo apt-get remove -y runc

sudo apt-get install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg \
    lsb-release

sudo groupadd docker
sudo usermod -aG docker $USER
newgrp docker 
pip3 install click

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit-base
curl https://get.docker.com | sh \
  && sudo systemctl --now enable docker
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
      && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
      && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
dockerd-rootless-setuptool.sh install
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker