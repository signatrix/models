FROM tensorflow/tensorflow:1.15.0-gpu-py3

# Install Object Detection API dependencies
# https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md
RUN ln -fs /usr/share/zoneinfo/Europe/Dublin /etc/localtime
RUN apt-get update && apt-get install -y --no-install-recommends \
        wget \
        git \
        python-lxml \
        python-tk \
        python-matplotlib \
        vim \
        python3-tk\
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
RUN pip install matplotlib==2.1.0 Cython contextlib2 jupyter beautifulsoup4 setuptools ruamel.yaml pykwalify scipy Pillow tf_slim

WORKDIR /tmp
RUN git clone https://github.com/cocodataset/cocoapi.git
RUN cd cocoapi/PythonAPI && make && cp -r pycocotools /tensorflow/
RUN rm -rf /tmp/cocoapi

WORKDIR /tensorflow/models/research
ENV PYTHONPATH=${PYTHONPATH}:/tensorflow/models/research:/tensorflow/models/research/slim

WORKDIR /tensorflow

RUN ["/bin/bash"]
