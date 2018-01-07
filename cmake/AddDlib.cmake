
configure_file(${DLIB_CFG}/dlib.cfg ${DLIB_PREFIX}/dlib-download/CMakeLists.txt)

execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${DLIB_PREFIX}/dlib-download)

if (result)
    message(FATAL_ERROR "CMake step for dlib failed: ${result}")
endif (result)

execute_process(COMMAND ${CMAKE_COMMAND} --build .
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${DLIB_PREFIX}/dlib-download)
if (result)
    message(FATAL_ERROR "Build step for dlib failed: ${result}")
endif (result)

add_subdirectory(${DLIB_PREFIX}/dlib-src/dlib ${CMAKE_BINARY_DIR}/dlib_build)