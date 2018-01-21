
cmake_minimum_required(VERSION 3.1)
project(kmeans)

set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

message("MODULE PATH: ${CMAKE_MODULE_PATH}")

find_package(xtensor)
find_package(xtensor-python)
find_package(xsimd)
find_package(NumPy)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
add_definitions("-DXTENSOR_USE_XSIMD")
include_directories(${xtensor_INCLUDES} ${xsimd_INCLUDES})
add_executable(kmeans_trainer main.cpp)

set(PYBIND11_PYTHON_VERSION 3)
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/pybind11/")
include_directories(../include ${xtensor_INCLUDE_DIRS} ${xtensor-python_INCLUDE_DIRS} ${xsimd_INCLUDE_DIRS} ${NUMPY_INCLUDE_DIRS})
pybind11_add_module(kmeans bindings.cpp)
