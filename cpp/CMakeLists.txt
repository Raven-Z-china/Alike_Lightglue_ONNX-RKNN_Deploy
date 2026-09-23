cmake_minimum_required(VERSION 3.10)
project(alike_lightglue_rknn CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall")

# --------------------------------------------------------------------------- #
# Paths are all cache variables - nothing is hard-coded to one machine.
#
#   cmake .. -DRKNN_API_ROOT=/path/to/rknn/runtime/Linux/librknn_api \
#            -DOPENCV_ROOT=/path/to/opencv4
#
# Cross-compiling for the board:
#   -DCMAKE_TOOLCHAIN_FILE=<aarch64-linux-gnu.cmake>
# The RKNN runtime ships aarch64 libs under `librknn_api/aarch64`, so the
# toolchain's sysroot must match the board's glibc (rk3588 images ship 2.34+).
# --------------------------------------------------------------------------- #
set(RKNN_API_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/deps/runtime/Linux/librknn_api"
    CACHE PATH "root of librknn_api")
set(OPENCV_ROOT "" CACHE PATH "root of an OpenCV 4 install (empty = use system)")

set(RKNN_INCLUDE_DIR "${RKNN_API_ROOT}/include")
set(RKNN_LIB_DIR "${RKNN_API_ROOT}/aarch64")
if(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
  # the simulator/host build is useful for compile-checking before flashing
  set(RKNN_LIB_DIR "${RKNN_API_ROOT}/x86_64")
endif()

# --------------------------------------------------------------------------- #
# Fail early with an actionable message rather than at link time.  A missing
# RKNN header is the most common setup error and its default error is opaque.
# --------------------------------------------------------------------------- #
if(NOT EXISTS "${RKNN_INCLUDE_DIR}/rknn_api.h")
  message(FATAL_ERROR
    "rknn_api.h not found under ${RKNN_INCLUDE_DIR}\n"
    "  Download the RKNN runtime from Rockchip (rknpu2) and either place it at\n"
    "  cpp/deps/runtime/Linux/librknn_api or pass -DRKNN_API_ROOT=<path>.\n"
    "  The header lives in <rknpu2>/runtime/Linux/librknn_api/include.")
endif()
if(NOT EXISTS "${RKNN_LIB_DIR}/librknnrt.so")
  message(FATAL_ERROR
    "librknnrt.so not found in ${RKNN_LIB_DIR}\n"
    "  Expected the aarch64 build for the board, or x86_64 for a host build.\n"
    "  Pass -DRKNN_LIB_DIR=<path> to override.")
endif()

if(OPENCV_ROOT)
  set(OpenCV_DIR "${OPENCV_ROOT}/lib/cmake/opencv4"
      CACHE PATH "OpenCV config dir" FORCE)
  if(NOT EXISTS "${OpenCV_DIR}")
    message(FATAL_ERROR "no OpenCV cmake config under ${OpenCV_DIR}")
  endif()
endif()
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)

# Two samples, same build.  `infer` takes one pair; `infer_sequence` matches one
# reference image against a folder of others, reusing the reference's extraction.
foreach(sample infer infer_sequence)
  add_executable(${sample} ${sample}.cpp)
  target_include_directories(${sample} PRIVATE "${RKNN_INCLUDE_DIR}")
  target_link_libraries(${sample} PRIVATE
    "${RKNN_LIB_DIR}/librknnrt.so"
    ${OpenCV_LIBS})

  # rpath so the binary finds librknnrt.so and the board's OpenCV without a
  # wrapper script - the usual cause of "works here, fails on the board".
  set_target_properties(${sample} PROPERTIES
    BUILD_RPATH "${RKNN_LIB_DIR};${OPENCV_ROOT}/lib"
    INSTALL_RPATH "${RKNN_LIB_DIR};${OPENCV_ROOT}/lib")
endforeach()

message(STATUS "RKNN headers : ${RKNN_INCLUDE_DIR}")
message(STATUS "RKNN libs    : ${RKNN_LIB_DIR}")
message(STATUS "OpenCV       : ${OpenCV_VERSION} (${OpenCV_DIR})")
