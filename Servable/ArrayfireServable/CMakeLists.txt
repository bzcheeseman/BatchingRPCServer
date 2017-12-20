cmake_minimum_required(VERSION 3.0)
project(BatchingRPCServer C CXX)

######### ARRAYFIRE #########

find_package(ArrayFire REQUIRED)
include_directories(${ArrayFire_INCLUDE_DIRS})

# If you intend to use OpenCL, you need to find it
find_package(OpenCL)
set(EXTRA_LIBS ${CMAKE_THREAD_LIBS_INIT} ${OpenCL_LIBRARIES})

# If cuda works, then uncomment this (and make sure you installed arrayfire with CUDA!)
find_package(CUDA)
find_package(NVVM) # this FIND script can be found in the ArrayFire CMake example repository
set(EXTRA_LIBS ${CMAKE_THREAD_LIBS_INIT} ${CUDA_LIBRARIES} ${NVVM_LIB})

######### ARRAYFIRE #########

file(GLOB servable_src src/*.cpp)
file(GLOB servable_include include/*.hpp)

add_library(af_servable ${servable_src} ${servable_include})
target_link_libraries(af_servable ${MXNET_LIBRARY})