cmake_minimum_required(VERSION 3.0)
project(BatchingRPCServer C CXX)

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

function(add_integration)
    if(NOT ARGN)
        message(SEND_ERROR "Error: No Libraries to link to! called without any proto files")
        return()
    endif()
    add_executable(TestIntegration ${CMAKE_CURRENT_SOURCE_DIR}/test/TestIntegration.cpp)
    target_link_libraries(TestIntegration gtest_main)
    foreach(LIB ${ARGN})
        find_library(LIB
                NAMES ${LIB}
                HINTS ${CMAKE_BINARY_DIR}/Servable/* ${CMAKE_BINARY_DIR}/Server
        )
        target_link_libraries(TestIntegration ${LIB})
    endforeach()

    add_test(NAME TestIntegration COMMAND TestIntegration)
endfunction()

add_subdirectory(Server)
add_subdirectory(Servable)

add_integration(MXNetServable TBServer)  # add more libs here as more components get added