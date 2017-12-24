FROM batchingrpcserver-env:latest

WORKDIR /BatchingRPCServer

ADD . .

RUN mkdir build && cd build && cmake .. && make check
