# GoogleTest ##########################################################################

configure_file(${GTEST_CFG}/gtest.cfg ${GTEST_PREFIX}/googletest-download/CMakeLists.txt)

message(STATUS "googletest prefix: ${GTEST_PREFIX}")

execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${GTEST_PREFIX}/googletest-download)
if(result)
    message(FATAL_ERROR "CMake step for googletest failed: ${result}")
endif()
execute_process(COMMAND ${CMAKE_COMMAND} --build .
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${GTEST_PREFIX}/googletest-download )
if(result)
    message(FATAL_ERROR "Build step for googletest failed: ${result}")
endif()

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

add_subdirectory(${GTEST_PREFIX}/googletest-src
        ${GTEST_PREFIX}/googletest-build
        EXCLUDE_FROM_ALL)

if (CMAKE_VERSION VERSION_LESS 2.8.11)
    include_directories("${gtest_SOURCE_DIR}/include")
endif()

# GoogleTest ##########################################################################

include_directories(${GTEST_PREFIX}/googletest-src/include)