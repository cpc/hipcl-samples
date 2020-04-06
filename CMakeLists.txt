# LLVM requires CMake >= 3.4.3
cmake_minimum_required(VERSION 3.4.3 FATAL_ERROR)
cmake_policy(VERSION 3.4)

project(HIPCL_SAMPLES
        VERSION "0.1"
        DESCRIPTION "HIPCL samples"
        LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_NO_SYSTEM_FROM_IMPORTED 1)

if(UNIX AND (CMAKE_CXX_COMPILER_ID STREQUAL "Clang"))
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 8.0.0)
    message(FATAL_ERROR "This project requires Clang >= 8.0")
  endif()
else()
  message(FATAL_ERROR "This project must be compiled with Clang")
endif()

message(STATUS "Using Clang: ${CMAKE_CXX_COMPILER}")
set(HIPCL_COMPILER "${CMAKE_CXX_COMPILER}")

enable_testing()

if(NOT HIPCL_PREFIX)
  set(HIPCL_PREFIX "/opt/hipcl")
endif()
find_package(HIP REQUIRED CONFIG PATHS "${HIPCL_PREFIX}")

if(NOT DEFINED INSTALL_BINDIR)
  set(INSTALL_BINDIR "${CMAKE_INSTALL_PREFIX}/bin" CACHE PATH "install destination" FORCE)
endif()

if(EXISTS "${CMAKE_SOURCE_DIR}/.git")
  set(DEFAULT_BUILD_TYPE "Debug")
else()
  set(DEFAULT_BUILD_TYPE "RelWithDebInfo")
endif()

if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "CMake build type not set, setting to ${DEFAULT_BUILD_TYPE}")
  set(CMAKE_BUILD_TYPE "${DEFAULT_BUILD_TYPE}" CACHE STRING "CMake build type" FORCE)
endif()

message(STATUS "CMake build type: ${CMAKE_BUILD_TYPE}")

set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
set(THREADS_PREFER_PTHREAD_FLAG TRUE)
find_package(Threads REQUIRED)
set(PTHREAD_LIBRARY Threads::Threads)

add_subdirectory(src)
