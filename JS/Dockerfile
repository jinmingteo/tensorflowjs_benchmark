# docker build . -t tensorflowjs
# xhost +local:docker && docker run --rm -e "DISPLAY=${DISPLAY}" --ipc=host -it --gpus all -v "/tmp/.X11-unix:/tmp/.X11-unix:rw" tensorflowjs

FROM nvcr.io/nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

# Install basic CLI tools etc.
RUN apt-get update && apt-get install -y --fix-missing --no-install-recommends \
        build-essential \
        curl \
        git-core \
        iputils-ping \
        pkg-config \
        rsync \
        software-properties-common \
        unzip \
        wget

# Install NodeJS
RUN curl --silent --location https://deb.nodesource.com/setup_10.x | bash -
RUN apt-get install --yes nodejs

WORKDIR /workspace

COPY ./package.json ./package-lock.json /workspace/
RUN npm ci --silent

COPY ./ /workspace

# Set working directory
CMD ["/bin/bash"]