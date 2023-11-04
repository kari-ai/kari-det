FROM nvcr.io/nvidia/cuda:11.7.0-cudnn8-devel-ubuntu20.04

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH
ENV OMP_NUM_THREADS=8

ARG DEBIAN_FRONTEND=noninteractive

# Install Dependencies of Miniconda
RUN apt-get update --fix-missing && \
    apt-get install -y wget bzip2 vim curl wget git ffmpeg libsm6 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install miniconda3
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc 
 
SHELL ["bin/bash", "--login", "-c"]
ADD kari-det.yaml . 
RUN conda update -n base -c defaults conda && \
    conda env create -f kari-det.yaml

RUN conda init bash

RUN conda activate kari-det

ADD kari-det.tar.gz .
