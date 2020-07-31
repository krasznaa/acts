
# Proper CUDA code handling requires at least CMake 3.10.
cmake_minimum_required(VERSION 3.10)

add_library(
  ActsCudaPlugin SHARED
  include/Acts/Plugins/Cuda/Utilities/Arrays.hpp
  include/Acts/Plugins/Cuda/Utilities/DeviceMatrix.cuh
  include/Acts/Plugins/Cuda/Utilities/DeviceMatrix.ipp
  include/Acts/Plugins/Cuda/Utilities/ErrorCheck.cuh
  include/Acts/Plugins/Cuda/Utilities/HostMatrix.hpp
  include/Acts/Plugins/Cuda/Utilities/StreamWrapper.hpp
  src/Utilities/Arrays.cu
  src/Utilities/ErrorCheck.cu
  src/Utilities/HostMatrix.cu
  src/Utilities/StreamHandlers.cuh
  src/Utilities/StreamHandlers.cu
  src/Utilities/StreamWrapper.cu
  )

target_include_directories(
  ActsCudaPlugin
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

target_link_libraries(
  ActsCudaPlugin
  PUBLIC ActsCore)

install(
  TARGETS ActsCudaPlugin
  EXPORT ActsCudaPluginTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(
  DIRECTORY include/Acts
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
