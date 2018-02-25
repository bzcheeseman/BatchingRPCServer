cmake_minimum_required(VERSION 3.0)
project(BatchingRPCServer C CXX)

set(CMAKE_CXX_STANDARD 11)

set(CMAKE_FIND_FRAMEWORK LAST)

find_package(Protobuf 3.5 REQUIRED)
find_package(GRPC 1.5 REQUIRED)
find_package(UUID 1.5 REQUIRED)

file(GLOB proto_files ${CMAKE_SOURCE_DIR}/proto/*.proto)

protobuf_generate_cpp(ProtoSources ProtoHeaders ${proto_files})
grpc_generate_cpp(GrpcSources GrpcHeaders ${CMAKE_CURRENT_BINARY_DIR} ${proto_files})

file(GLOB batching_server_src ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
file(GLOB batching_server_include ${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp)

add_library(TBServer SHARED
        ${batching_server_src} ${batching_server_include}
        ${ProtoSources} ${ProtoHeaders}
        ${GrpcSources} ${GrpcHeaders}
)
target_link_libraries(TBServer
        PUBLIC c++
        PUBLIC gRPC::grpc
        PUBLIC gRPC::grpc++
        PUBLIC ${PROTOBUF_LIBRARIES}
        PUBLIC ${UUID_LIBRARY}
)
target_include_directories(TBServer
        PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
        PUBLIC ${CMAKE_CURRENT_BINARY_DIR}
        PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../Servable
        PUBLIC ${UUID_INCLUDE_DIRS}
)

add_gtest(TBServer TBServer)

# Add my specific sources (not generated)
set(SOURCES ${batching_server_src} ${batching_server_include} ${SOURCES} PARENT_SCOPE)
# Add my specific libraries (not generated)
set(LIBS ${UUID_LIBRARIES} ${LIBS} PARENT_SCOPE)
# Add my specific include dirs (not generated)
set(INCLUDE_DIRS ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${UUID_INCLUDE_DIRS} ${INCLUDE_DIRS} PARENT_SCOPE)
