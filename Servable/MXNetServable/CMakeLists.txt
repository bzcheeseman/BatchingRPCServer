cmake_minimum_required(VERSION 3.5)
project(rpc_batch_scheduler C CXX)

######### MXNET #########
find_package(MXNet REQUIRED)
######### MXNET #########

file(GLOB servable_src src/*.cpp)
file(GLOB servable_include include/*.hpp)

find_package(protobuf 3.5 REQUIRED)

file(GLOB proto_files ${CMAKE_SOURCE_DIR}/proto/*.proto)

protobuf_generate_cpp(ProtoSources ProtoHeaders ${proto_files})

add_library(mx_servable SHARED
        ${servable_src} ${servable_include}
        ${CMAKE_CURRENT_SOURCE_DIR}/../Servable.hpp
        ${ProtoSources} ${ProtoHeaders}
)
target_link_libraries(mx_servable PUBLIC ${MXNET_LIBS} PUBLIC ${PROTOBUF_LIBRARIES})
target_include_directories(mx_servable
        PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${MXNET_INCLUDE_DIRS}
               ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/..
)

add_gtest(MXNetServable mx_servable)
