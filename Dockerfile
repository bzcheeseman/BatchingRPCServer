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
 uuid-dev \
 autoconf \
 automake \
 libtool \
 curl \
 make \
 g++ \
 unzip \
 shtool \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

ADD "https://cmake.org/files/v3.10/cmake-3.10.0-Linux-x86_64.sh" "/cmake-3.10.0-Linux-x86_64.sh"
RUN mkdir /opt/cmake &&\
        sh /cmake-3.10.0-Linux-x86_64.sh --prefix=/opt/cmake --skip-license &&\
        ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake


RUN ln -s /usr/local/cuda-9.0/targets/x86_64-linux/lib/stubs/libcuda.so /usr/local/cuda-9.0/targets/x86_64-linux/lib/stubs/libcuda.so.1

WORKDIR /opt

RUN git clone --recursive --branch 0.12.0 --single-branch https://github.com/apache/incubator-mxnet.git mxnet
RUN git clone --branch v3.5.0 --single-branch https://github.com/google/protobuf.git
RUN git clone --recursive --branch v1.8.2 --single-branch https://github.com/grpc/grpc.git

# ------ MXNet ------
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
# ------ MXNet ------

# ------ Protobuf ------
WORKDIR /opt/protobuf

RUN ./autogen.sh && ./configure --prefix=/usr && make && make check && make install
# ------ Protobuf ------

# ------ gRPC ------
WORKDIR /opt/grpc

RUN make && make install
# ------ gRPC ------

RUN ldconfig

WORKDIR /BatchingRPCServer

ADD . .

RUN mkdir build && cd build && cmake .. && make check
