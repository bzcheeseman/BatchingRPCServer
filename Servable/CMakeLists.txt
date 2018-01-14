cmake_minimum_required(VERSION 3.0)
project(BatchingRPCServer C CXX)

# Add subdirectory first to populate SOURCES, LIBS, INCLUDE_DIRS
add_subdirectory(MXNetServable)
add_subdirectory(DlibServable)  # Not ready yet

# Add this level's SOURCES, INCLUDE_DIRS and forward them and LIBS to upper scope
set(SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/Servable.hpp ${SOURCES} PARENT_SCOPE)
set(LIBS ${LIBS} PARENT_SCOPE)
set(INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR} ${INCLUDE_DIRS} PARENT_SCOPE)