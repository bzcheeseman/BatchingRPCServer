cmake_minimum_required(VERSION 3.0)
project(rpc_batch_scheduler C CXX)

set(CMAKE_CXX_STANDARD 11)

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

add_subdirectory(${CMAKE_SOURCE_DIR}/BatchingServer)