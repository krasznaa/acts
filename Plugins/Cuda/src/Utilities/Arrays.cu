// This file is part of the Acts project.
//
// Copyright (C) 2020 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

// CUDA plugin include(s).
#include "Acts/Plugins/Cuda/Seeding/Types.hpp"
#include "Acts/Plugins/Cuda/Utilities/Arrays.hpp"
#include "Acts/Plugins/Cuda/Utilities/ErrorCheck.cuh"
#include "Acts/Plugins/Cuda/Utilities/MemoryManager.hpp"

// CUDA include(s).
#include <cuda_runtime.h>

namespace Acts {
namespace Cuda {
namespace details {

void DeviceArrayDeleter::operator()(void* ptr) {
  // Do not do anything! Memory will be "given up" using a call to
  // @c Acts::Cuda::MemoryManager::reset().
  return;
}

void HostArrayDeleter::operator()(void* ptr) {
  // Ignore null-pointers.
  if (ptr == nullptr) {
    return;
  }

  // Free the pinned host memory.
  ACTS_CUDA_ERROR_CHECK(cudaFreeHost(ptr));
  return;
}

}  // namespace details

template <typename T>
device_array<T> make_device_array(std::size_t size) {
  // Allocate the memory.
  T* ptr = nullptr;
  if (size != 0) {
    ptr = reinterpret_cast<T*>(
        MemoryManager::instance().allocate(size * sizeof(T)));
  }
  // Create the smart pointer.
  return device_array<T>(ptr);
}

template <typename T>
host_array<T> make_host_array(std::size_t size) {
  // Allocate the memory.
  T* ptr = nullptr;
  if (size != 0) {
    ACTS_CUDA_ERROR_CHECK(cudaMallocHost(&ptr, size * sizeof(T)));
  }
  // Create the smart pointer.
  return host_array<T>(ptr);
}

template <typename T>
void copyToDevice(device_array<T>& dev, const host_array<T>& host,
                  std::size_t arraySize) {
  ACTS_CUDA_ERROR_CHECK(cudaMemcpy(dev.get(), host.get(), arraySize * sizeof(T),
                                   cudaMemcpyHostToDevice));
  return;
}

template <typename T>
void copyToHost(host_array<T>& host, const device_array<T>& dev,
                std::size_t arraySize) {
  ACTS_CUDA_ERROR_CHECK(cudaMemcpy(host.get(), dev.get(), arraySize * sizeof(T),
                                   cudaMemcpyDeviceToHost));
  return;
}

}  // namespace Cuda
}  // namespace Acts

/// Helper macro for instantiating the template code for a given type
///
/// Note that nvcc (at least as of CUDA version 11.0.2) does not allow us to
/// instantiate our custom unique pointer types through their typedef'd names.
/// That's why the following expressions are as long as they are.
///
#define INST_ARRAY_FOR_TYPE(TYPE)                                              \
  template class std::unique_ptr<TYPE,                                         \
                                 Acts::Cuda::details::DeviceArrayDeleter>;     \
  template std::unique_ptr<TYPE, Acts::Cuda::details::DeviceArrayDeleter>      \
      Acts::Cuda::make_device_array<TYPE>(std::size_t);                        \
  template class std::unique_ptr<TYPE, Acts::Cuda::details::HostArrayDeleter>; \
  template std::unique_ptr<TYPE, Acts::Cuda::details::HostArrayDeleter>        \
      Acts::Cuda::make_host_array<TYPE>(std::size_t);                          \
  template void                                                                \
  Acts::Cuda::copyToDevice<TYPE>(std::unique_ptr<TYPE,                         \
      Acts::Cuda::details::DeviceArrayDeleter>&,                               \
      const std::unique_ptr<TYPE, Acts::Cuda::details::HostArrayDeleter>&,     \
      std::size_t);                                                            \
  template void                                                                \
  Acts::Cuda::copyToHost<TYPE>(std::unique_ptr<TYPE,                           \
      Acts::Cuda::details::HostArrayDeleter>&,                                 \
      const std::unique_ptr<TYPE, Acts::Cuda::details::DeviceArrayDeleter>&,   \
      std::size_t)

// Instantiate the templated functions for all primitive types.
INST_ARRAY_FOR_TYPE(char);
INST_ARRAY_FOR_TYPE(unsigned char);
INST_ARRAY_FOR_TYPE(short);
INST_ARRAY_FOR_TYPE(unsigned short);
INST_ARRAY_FOR_TYPE(int);
INST_ARRAY_FOR_TYPE(unsigned int);
INST_ARRAY_FOR_TYPE(long);
INST_ARRAY_FOR_TYPE(unsigned long);
INST_ARRAY_FOR_TYPE(long long);
INST_ARRAY_FOR_TYPE(unsigned long long);
INST_ARRAY_FOR_TYPE(float);
INST_ARRAY_FOR_TYPE(double);

// Instantiate them for any necessary custom type(s) as well.
INST_ARRAY_FOR_TYPE(Acts::Cuda::details::Triplet);
INST_ARRAY_FOR_TYPE(Acts::Cuda::details::SpacePoint);
INST_ARRAY_FOR_TYPE(Acts::Cuda::details::DubletCounts);
INST_ARRAY_FOR_TYPE(Acts::Cuda::details::LinCircle);

// Clean up.
#undef INST_ARRAY_FOR_TYPE
