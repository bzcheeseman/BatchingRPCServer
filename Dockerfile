FROM nvidia/cuda:9.0-cudnn7-devel

RUN apt-get update -y \
 && apt-get install -y --no-install-recommends \
 git \
 wget \
 build-essential \
 libopencv-dev \
 libopenblas-dev \
 google-perftools \
 ca-certificates \
 python-pip \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/local/cuda-9.0/targets/x86_64-linux/lib/stubs/libcuda.so /usr/local/cuda-9.0/targets/x86_64-linux/lib/stubs/libcuda.so.1

WORKDIR /opt

RUN git clone --recursive --branch 0.12.0 --single-branch https://github.com/apache/incubator-mxnet.git mxnet

WORKDIR /opt/mxnet

RUN make -j4 USE_CPP_PACKAGE=1 USE_CUDA_PATH=/usr/local/cuda USE_CUDA=1 USE_CUDNN=1 USE_BLAS=openblas USE_GPERFTOOLS=1

RUN ln -s /opt/mxnet/lib/libmxnet.so /usr/lib; \
    mkdir /usr/include/mxnet; \
    ln -s include /usr/include/mxnet/; \
    mkdir /usr/include/mxnet/nnvm; \
    ln -s nnvm/include /usr/include/mxnet/nnvm/; \
    mkdir /usr/include/mxnet/dmlc-core; \
    ln -s dmlc-core/include /usr/include/mxnet/dmlc-core; \
    mkdir /usr/include/mxnet/cpp-package; \
    ln -s cpp-package/include /usr/include/mxnet/cpp-package/include

RUN rm /usr/local/cuda-9.0/targets/x86_64-linux/lib/stubs/libcuda.so.1

RUN ldconfig

WORKDIR /BatchingRPCServer

ADD . .

RUN mkdir build && cd build && cmake .. && make check
