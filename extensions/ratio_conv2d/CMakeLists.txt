CMAKE_MINIMUM_REQUIRED ( VERSION 3.8.0 )

# THIS HAS TO COME BEFORE THE PROJECT LINE
set(CMAKE_CXX_COMPILER "g++")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_COMPILER nvcc)
set(CMAKE_CUDA_STANDARD 14)
cmake_policy(SET CMP0104 NEW)

PROJECT(ratio_conv2d LANGUAGES CXX CUDA)

set (VERSION_MAJOR 1)
set (VERSION_MINOR 0)

if ("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
  message(SEND_ERROR "In-source builds are not allowed.")
endif ()

set(CMAKE_VERBOSE_MAKEFILE OFF)
set(CMAKE_COLOR_MAKEFILE   ON)

SET (SRC_DIR "./src")
SET (MY_LIB_PATH "${CMAKE_CURRENT_LIST_DIR}/lib/")
SET (MY_BIN_PATH "${CMAKE_CURRENT_LIST_DIR}/bin/")
include_directories(${SRC_DIR})

##-----------compiler settings---------------------------
SET(CMAKE_CONFIGURATION_TYPES "RelWithDebInfo;Debug" CACHE STRING "RelWithDebInfo;Debug" FORCE)
SET(CMAKE_BUILD_TYPE "RelWithDebInfo")
#SET(CMAKE_BUILD_TYPE "Debug")
#set(CMAKE_CUDA_FLAGS "-gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_61,code=sm_61") #it will fetch automatically for the current GPU
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "60;61")
endif()
SET(MAKE_CUDA_SEPARABLE_COMPILATION ON)
SET(BUILD_SHARED_LIBS OFF)
include(${CMAKE_CURRENT_LIST_DIR}/cmake/compiler_settings.cmake.txt)
include(${CMAKE_CURRENT_LIST_DIR}/cmake/functions.cmake.txt)
##-------------------------
#clear_flag(CMAKE_CUDA_FLAGS "-lineinfo")
#add_flags(CMAKE_CUDA_FLAGS "-Xptxas=--verbose")
add_flags(CMAKE_CUDA_FLAGS "-src-in-ptx")
add_flags(CMAKE_CUDA_FLAGS "--use_fast_math")
add_flags(CMAKE_CUDA_FLAGS "--restrict")
add_flags(CMAKE_CUDA_FLAGS "-Xcompiler=-fPIC")
add_flags(CMAKE_CXX_FLAGS "-fPIC")
message(STATUS "CMAKE_CUDA_FLAGS= ${CMAKE_CUDA_FLAGS}")
##-------------------------                                      
#----------print includes--------------------------------
message("INCLUDE:")
get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
foreach(dir ${dirs})
  message(STATUS "dir='${dir}'")
endforeach()
##--------------------------------------------------------
SET(SRC
${SRC_DIR}/ratio_conv2d.cu
${SRC_DIR}/ndarray/error.cpp
${SRC_DIR}/ndarray/type_name.cpp
${SRC_DIR}/ndarray/ndarray_mem.cpp
${SRC_DIR}/ndarray/ndarray.cpp
${SRC_DIR}/ndarray/ndarray_op_cuda.cu
${SRC_DIR}/ndarray/ndarray_op.cpp
#${SRC_DIR}/ndarray/test_transform.cpp
)

my_cuda_add_library(ratio_conv2d ${SRC})

my_cuda_add_executable(test_ratio_conv2d ${SRC_DIR}/ratio_conv2d_test.cpp)
