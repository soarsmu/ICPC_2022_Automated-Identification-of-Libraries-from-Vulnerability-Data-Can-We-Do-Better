#
# Copyright (c) 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
#

cmake_minimum_required(VERSION 2.8.9)
project(extremetext)

# The version number.
set (extremetext_VERSION_MAJOR 0)
set (extremetext_VERSION_MINOR 1)

include_directories(extremetext)

set(CMAKE_CXX_FLAGS " -pthread -std=c++11 -funroll-loops -O3 -march=native")

set(HEADER_FILES
    src/args.h
    src/dictionary.h
    src/fasttext.h
    src/kmeans.h
    src/loss_ensemble.h
    src/loss_plt.h
    src/loss_sigmoid.h
    src/losslayer.h
    src/matrix.h
    src/model.h
    src/productquantizer.h
    src/qmatrix.h
    src/real.h
    src/smatrix.h
    src/threads.h
    src/utils.h
    src/vector.h)

set(SOURCE_FILES
    src/args.cc
    src/dictionary.cc
    src/fasttext.cc
    src/kmeans.cc
    src/loss_ensemble.cc
    src/loss_plt.cc
    src/loss_sigmoid.cc
    src/losslayer.cc
    src/main.cc
    src/matrix.cc
    src/model.cc
    src/productquantizer.cc
    src/qmatrix.cc
    src/utils.cc
    src/vector.cc)

add_library(extremetext-shared SHARED ${SOURCE_FILES} ${HEADER_FILES})
add_library(extremetext-static STATIC ${SOURCE_FILES} ${HEADER_FILES})
add_library(extremetext-static_pic STATIC ${SOURCE_FILES} ${HEADER_FILES})
set_target_properties(extremetext-shared PROPERTIES OUTPUT_NAME extremetext)
set_target_properties(extremetext-static PROPERTIES OUTPUT_NAME extremetext)
set_target_properties(extremetext-static_pic PROPERTIES OUTPUT_NAME extremetext_pic
    POSITION_INDEPENDENT_CODE True)
add_executable(extremetext-bin src/main.cc)
target_link_libraries(extremetext-bin pthread extremetext-static)
set_target_properties(extremetext-bin PROPERTIES PUBLIC_HEADER "${HEADER_FILES}" OUTPUT_NAME extremetext)
install (TARGETS extremetext-shared
    LIBRARY DESTINATION lib)
install (TARGETS extremetext-static
    ARCHIVE DESTINATION lib)
install (TARGETS extremetext-static_pic
    ARCHIVE DESTINATION lib)
install (TARGETS extremetext-bin
    RUNTIME DESTINATION bin
    PUBLIC_HEADER DESTINATION include/extremetext)
