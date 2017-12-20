cmake_minimum_required(VERSION 3.0)
project(rpc_batch_scheduler C CXX)

set(CMAKE_CXX_STANDARD 11)

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

set(GTEST_CFG ${CMAKE_MODULE_PATH})
set(GTEST_PREFIX ${CMAKE_CURRENT_BINARY_DIR})
include(AddGTest)

macro(add_gtest test_name lib)
    add_executable(Test${test_name} ${CMAKE_CURRENT_SOURCE_DIR}/test/Test${test_name}.cpp)
    target_link_libraries(Test${test_name} gtest_main ${lib})
    add_test(NAME Test${test_name} COMMAND Test${test_name})
endmacro()

add_subdirectory(Server)
add_subdirectory(Servable)