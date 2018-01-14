cmake_minimum_required(VERSION 3.5)
project(BatchingRPCServer C CXX)

######### MXNET #########
find_package(MXNet REQUIRED)
######### MXNET #########

file(GLOB servable_src ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
file(GLOB servable_include ${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp)

find_package(Protobuf 3.5 REQUIRED)

file(GLOB proto_files ${CMAKE_SOURCE_DIR}/proto/*.proto)

protobuf_generate_cpp(ProtoSources ProtoHeaders ${proto_files})

add_library(MXNetServable SHARED
        ${servable_src} ${servable_include}
        ${CMAKE_CURRENT_SOURCE_DIR}/../Servable.hpp
        ${ProtoSources} ${ProtoHeaders}
)
target_link_libraries(MXNetServable
        PUBLIC ${MXNET_LIBS}
        PUBLIC ${PROTOBUF_LIBRARIES}
        PUBLIC c++
)
target_include_directories(MXNetServable
        PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
        PUBLIC ${MXNET_INCLUDE_DIRS}
        PUBLIC ${CMAKE_CURRENT_BINARY_DIR}
        PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..
)

add_gtest(MXNetServable MXNetServable)

# Add my source files to the upper level sources (not generated)
set(SOURCES ${servable_src} ${servable_include} ${SOURCES} PARENT_SCOPE)
# Add libs specific to me here
set(LIBS ${MXNET_LIBS} ${LIBS} PARENT_SCOPE)
# Add specific include directories here (not generated)
set(INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include ${MXNET_INCLUDE_DIRS} ${INCLUDE_DIRS} PARENT_SCOPE)