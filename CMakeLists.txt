cmake_minimum_required(VERSION 3.0)
project(BatchingRPCServer C CXX)

set(CMAKE_CXX_STANDARD 11)

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

set(GTEST_CFG ${CMAKE_MODULE_PATH})
set(GTEST_PREFIX ${CMAKE_CURRENT_BINARY_DIR})
include(AddGTest)

enable_testing()

add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND} -V)

function(add_gtest test_name lib)
    add_executable(Test${test_name} EXCLUDE_FROM_ALL ${CMAKE_CURRENT_SOURCE_DIR}/test/Test${test_name}.cpp)
    target_link_libraries(Test${test_name} gtest_main ${lib})
    add_test(NAME Test${test_name} COMMAND ${CMAKE_CURRENT_BINARY_DIR}/Test${test_name})
    add_dependencies(check Test${test_name})
endfunction()

function(add_integration suffix)
    if(NOT ARGN)
        message(SEND_ERROR "Error: No Libraries to link to!")
        return()
    endif()
    add_executable(TestIntegration${suffix} EXCLUDE_FROM_ALL ${CMAKE_CURRENT_SOURCE_DIR}/test/TestIntegrationMXNet.cpp)
    target_link_libraries(TestIntegration${suffix} gtest_main)
    foreach(LIB ${ARGN})
        find_library(LIB
                NAMES ${LIB}
                HINTS ${CMAKE_BINARY_DIR}/Servable/* ${CMAKE_BINARY_DIR}/Server
        )
        target_link_libraries(TestIntegration${suffix} ${LIB})
    endforeach()

    add_test(NAME TestIntegration${suffix} COMMAND ${CMAKE_CURRENT_BINARY_DIR}/TestIntegration${suffix})
    add_dependencies(check TestIntegration${suffix})
endfunction()

add_subdirectory(Server)
add_subdirectory(Servable)

find_package(protobuf 3.5 REQUIRED)
find_package(GRPC 1.5 REQUIRED)

# Generate my proto files
file(GLOB proto_files ${CMAKE_SOURCE_DIR}/proto/*.proto)
protobuf_generate_cpp(ProtoSources ProtoHeaders ${proto_files})
grpc_generate_cpp(GrpcSources GrpcHeaders ${CMAKE_CURRENT_BINARY_DIR} ${proto_files})

# Add the generated stuff and libs that everyone needs here
set(SOURCES ${ProtoSources} ${ProtoHeaders} ${GrpcSources} ${GrpcHeaders} ${SOURCES})
set(LIBS gRPC::grpc gRPC::grpc++ ${PROTOBUF_LIBRARIES} ${LIBS})
set(INCLUDE_DIRS ${CMAKE_CURRENT_BINARY_DIR} ${INCLUDE_DIRS})

add_library(BatchingRPCServer SHARED ${SOURCES})
target_link_libraries(BatchingRPCServer PUBLIC ${LIBS})
target_include_directories(BatchingRPCServer PUBLIC ${INCLUDE_DIRS})

add_integration(MXNet BatchingRPCServer)

# Add cmake target so we can also call make doc to create the documentation
add_custom_target(doc
        COMMAND doxygen ./Doxyfile
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/docs
)

file(GLOB_RECURSE
        ALL_CXX_SOURCE_FILES
        Servable/*.[chi]pp Servable/*.[chi]xx Servable/*.cc Servable/*.hh Servable/*.ii Servable/*.[CHI]
        Server/*.[chi]pp Server/*.[chi]xx Server/*.cc Server/*.hh Server/*.ii Server/*.[CHI]
        test/*.[chi]pp test/*.[chi]xx test/*.cc test/*.hh test/*.ii test/*.[CHI]
)

find_program(CLANG_FORMAT "clang-format")
if(CLANG_FORMAT)
    add_custom_target(
            clang-format
            COMMAND ${CLANG_FORMAT}
            -i
            -style=file
            ${ALL_CXX_SOURCE_FILES}

    )
endif()
