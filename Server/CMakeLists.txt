cmake_minimum_required(VERSION 3.0)
project(rpc_batch_scheduler C CXX)

set(CMAKE_FIND_FRAMEWORK LAST)

find_package(protobuf 3.5 REQUIRED)
find_package(GRPC 1.5 REQUIRED)
find_package(UUID 1.5 REQUIRED)

file(GLOB proto_files ${CMAKE_SOURCE_DIR}/proto/*.proto)

protobuf_generate_cpp(ProtoSources ProtoHeaders ${proto_files})
grpc_generate_cpp(GrpcSources GrpcHeaders ${CMAKE_CURRENT_BINARY_DIR} ${proto_files})

file(GLOB batching_server_src src/*.cpp)
file(GLOB batching_server_include include/*.hpp)

add_library(batching_server SHARED
        ${batching_server_src} ${batching_server_include}
        ${ProtoSources} ${ProtoHeaders}
        ${GrpcSources} ${GrpcHeaders}
)
target_link_libraries(batching_server gRPC::grpc gRPC::grpc++ ${PROTOBUF_LIBRARIES})
target_include_directories(batching_server
        PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_BINARY_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../Servable ${UUID_INCLUDE_DIRS}
)

add_gtest(TensorBatchingServer batching_server)
