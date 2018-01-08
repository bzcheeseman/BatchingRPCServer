# - Try to find MXNet
# Once done this will define
#
#  MXNET_FOUND - system has MXNET
#  MXNET_INCLUDE_DIRS - the MXNET include directory
#  MXNET_LIBS - Link these to use MXNET
#
# Copyright (c) 2017 Aman LaChapelle
# Full license at BatchingRPCServer/LICENSE.txt
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


find_path(MXNET_INCLUDE
        NAMES mxnet/c_api.h
        PATHS
        $ENV{MXNET_ROOT}/include
)

find_path(MXNET_NNVM_INCLUDE
        NAMES nnvm/c_api.h
        PATHS
        $ENV{MXNET_ROOT}/nnvm/include
)

find_path(MXNET_DMLC_INCLUDE
        NAMES dmlc/any.h
        PATHS
        $ENV{MXNET_ROOT}/dmlc-core/include
)

find_path(MXNET_CPP_INCLUDE
        NAMES mxnet-cpp/MxNetCpp.h
        PATHS
        $ENV{MXNET_ROOT}/cpp-package/include
)

list(APPEND MXNET_INCLUDE_DIRS ${MXNET_INCLUDE})
list(APPEND MXNET_INCLUDE_DIRS ${MXNET_NNVM_INCLUDE})
list(APPEND MXNET_INCLUDE_DIRS ${MXNET_DMLC_INCLUDE})
list(APPEND MXNET_INCLUDE_DIRS ${MXNET_CPP_INCLUDE})

find_library(MXNET_LIBRARY
        NAMES mxnet
        PATHS $ENV{MXNET_ROOT}/lib /usr/lib/
)

list(APPEND MXNET_LIBS ${MXNET_LIBRARY})

set(MXNET_FOUND TRUE)

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(MXNet DEFAULT_MSG MXNET_LIBRARY MXNET_INCLUDE_DIRS)
mark_as_advanced(MXNET_INCLUDE_DIRS MXNET_LIBS)

