// This file is part of the Acts project.
//
// Copyright (C) 2020 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

// CUDA plugin include(s).
#include "Acts/Plugins/Cuda/Utilities/Arrays.hpp"
#include "Acts/Plugins/Cuda/Utilities/ErrorCheck.cuh"
#include "Acts/Plugins/Cuda/Utilities/HostMatrix.hpp"
#include "Acts/Plugins/Cuda/Utilities/MatrixMacros.hpp"

// Boost include(s).
#include <boost/test/unit_test.hpp>

// CUDA include(s).
#include <cuda_runtime.h>

// System include(s).
#include <cmath>

namespace Acts {
namespace Cuda {
namespace Test {

/// A 32-bit representation of Pi, for better numerical comparisons.
static constexpr float PI = M_PI;

/// Simple kernels performing matrix multiplication.
/// @{
__global__ void matrixMultiply(std::size_t size, float* array,
                               float multiplier) {
  // Get the index to work on.
  const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= size) {
    return;
  }

  // Perform the multiplication.
  array[index] *= multiplier;
  return;
}

__global__ void matrixMultiply(std::size_t x_size, std::size_t y_size,
                               float* array, float multiplier) {
  // Get the index to work on.
  const std::size_t x_index = blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t y_index = blockIdx.y * blockDim.y + threadIdx.y;
  if ((x_index >= x_size) || (y_index >= y_size)) {
    return;
  }

  // Perform the multiplication.
  ACTS_CUDA_MATRIX2D_ELEMENT(array, x_size, y_size, x_index, y_index) *=
      multiplier;
  return;
}

__global__ void matrixMultiply(std::size_t x_size, std::size_t y_size,
                               std::size_t z_size, float* array,
                               float multiplier) {
  // Get the index to work on.
  const std::size_t x_index = blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t y_index = blockIdx.y * blockDim.y + threadIdx.y;
  const std::size_t z_index = blockIdx.z * blockDim.z + threadIdx.z;
  if ((x_index >= x_size) || (y_index >= y_size) || (z_index >= z_size)) {
    return;
  }

  // Perform the multiplication.
  ACTS_CUDA_MATRIX3D_ELEMENT(array, x_size, y_size, z_size, x_index, y_index,
                             z_index) *= multiplier;
  return;
}
/// @}

BOOST_AUTO_TEST_SUITE(Utilities)
BOOST_AUTO_TEST_CASE(Matrix1D) {
  // Create a 1-dimensional matrix on the host.
  static constexpr std::size_t X_SIZE = 1000;
  HostMatrix<1, float> hostMatrix({X_SIZE});
  BOOST_TEST_REQUIRE(hostMatrix.totalSize() == X_SIZE);
  BOOST_TEST_REQUIRE(hostMatrix.size().at(0) == X_SIZE);
  for (std::size_t i = 0; i < X_SIZE; ++i) {
    hostMatrix.set({i}, i * PI);
  }

  // Copy the underlying memory block to the device.
  auto deviceMatrix = make_device_array<float>(hostMatrix.totalSize());
  hostMatrix.copyTo(deviceMatrix);

  // Perform a matrix multiplication on the GPU.
  static constexpr int blockSize = 256;
  static const int numBlocks = (hostMatrix.size().at(0) + blockSize -
                                1) / blockSize;
  matrixMultiply<<<numBlocks, blockSize>>>(
      hostMatrix.size().at(0), deviceMatrix.get(), 2.0f);
  ACTS_CUDA_ERROR_CHECK(cudaGetLastError());
  ACTS_CUDA_ERROR_CHECK(cudaDeviceSynchronize());

  // Check the results.
  hostMatrix.copyFrom(deviceMatrix);
  float maxDeviation = 0.0f;
  for (std::size_t i = 0; i < X_SIZE; ++i) {
    maxDeviation = std::max(maxDeviation, std::abs(hostMatrix.get({i}) -
                                                   i * PI * 2.0f));
  }
  BOOST_TEST_REQUIRE(maxDeviation < 0.001);
}

BOOST_AUTO_TEST_CASE(Matrix2D) {
  // Create a 2-dimensional matrix on the host.
  static constexpr std::size_t X_SIZE = 1000;
  static constexpr std::size_t Y_SIZE = 800;
  HostMatrix<2, float> hostMatrix({X_SIZE, Y_SIZE});
  BOOST_TEST_REQUIRE(hostMatrix.totalSize() == X_SIZE * Y_SIZE);
  BOOST_TEST_REQUIRE(hostMatrix.size().at(0) == X_SIZE);
  BOOST_TEST_REQUIRE(hostMatrix.size().at(1) == Y_SIZE);
  for (std::size_t i = 0; i < X_SIZE; ++i) {
    for (std::size_t j = 0; j < Y_SIZE; ++j) {
      hostMatrix.set({i, j}, i * j * PI);
    }
  }

  // Copy the underlying memory block to the device.
  auto deviceMatrix = make_device_array<float>(hostMatrix.totalSize());
  hostMatrix.copyTo(deviceMatrix);

  // Perform a matrix multiplication on the GPU.
  static constexpr dim3 blockSize(16, 16);
  static const dim3 numBlocks(
      (hostMatrix.size().at(0) + blockSize.x - 1) / blockSize.x,
      (hostMatrix.size().at(1) + blockSize.y - 1) / blockSize.y);
  matrixMultiply<<<numBlocks, blockSize>>>(
      hostMatrix.size().at(0), hostMatrix.size().at(1),
      deviceMatrix.get(), 5.0f);
  ACTS_CUDA_ERROR_CHECK(cudaGetLastError());
  ACTS_CUDA_ERROR_CHECK(cudaDeviceSynchronize());

  // Check the results.
  hostMatrix.copyFrom(deviceMatrix);
  float maxDeviation = 0.0f;
  for (std::size_t i = 0; i < X_SIZE; ++i) {
    for (std::size_t j = 0; j < Y_SIZE; ++j) {
      maxDeviation = std::max(maxDeviation, std::abs(hostMatrix.get({i, j}) -
                                                     i * j * PI * 5.0f));
    }
  }
  BOOST_TEST_REQUIRE(maxDeviation < 0.001);
}

BOOST_AUTO_TEST_CASE(Matrix3D) {
  // Create a 3-dimensional matrix on the host. Note that this test needs
  // ~1.6 GB of memory on the GPU. So old GPUs may have issues with it...
  static constexpr std::size_t X_SIZE = 100;
  static constexpr std::size_t Y_SIZE = 80;
  static constexpr std::size_t Z_SIZE = 500;
  HostMatrix<3, float> hostMatrix({X_SIZE, Y_SIZE, Z_SIZE});
  BOOST_TEST_REQUIRE(hostMatrix.totalSize() == X_SIZE * Y_SIZE * Z_SIZE);
  BOOST_TEST_REQUIRE(hostMatrix.size().at(0) == X_SIZE);
  BOOST_TEST_REQUIRE(hostMatrix.size().at(1) == Y_SIZE);
  BOOST_TEST_REQUIRE(hostMatrix.size().at(2) == Z_SIZE);
  for (std::size_t i = 0; i < X_SIZE; ++i) {
    for (std::size_t j = 0; j < Y_SIZE; ++j) {
      for (std::size_t k = 0; k < Z_SIZE; ++k) {
        hostMatrix.set({i, j, k}, i * j * k * PI);
      }
    }
  }

  // Copy the underlying memory block to the device.
  auto deviceMatrix = make_device_array<float>(hostMatrix.totalSize());
  hostMatrix.copyTo(deviceMatrix);

  // Perform a matrix multiplication on the GPU.
  static constexpr dim3 blockSize(8, 8, 4);
  static const dim3 numBlocks(
      (hostMatrix.size().at(0) + blockSize.x - 1) / blockSize.x,
      (hostMatrix.size().at(1) + blockSize.y - 1) / blockSize.y,
      (hostMatrix.size().at(2) + blockSize.z - 1) / blockSize.z);
  matrixMultiply<<<numBlocks, blockSize>>>(
      hostMatrix.size().at(0), hostMatrix.size().at(1), hostMatrix.size().at(2),
      deviceMatrix.get(), 1.2f);
  ACTS_CUDA_ERROR_CHECK(cudaGetLastError());
  ACTS_CUDA_ERROR_CHECK(cudaDeviceSynchronize());

  // Check the results.
  hostMatrix.copyFrom(deviceMatrix);
  float maxDeviation = 0.0f;
  for (std::size_t i = 0; i < X_SIZE; ++i) {
    for (std::size_t j = 0; j < Y_SIZE; ++j) {
      for (std::size_t k = 0; k < Z_SIZE; ++k) {
        maxDeviation = std::max(maxDeviation,
                                std::abs(hostMatrix.get({i, j, k}) -
                                         i * j * k * PI * 1.2f));
      }
    }
  }
  BOOST_TEST_REQUIRE(maxDeviation < 0.001);
}
BOOST_AUTO_TEST_SUITE_END()

}  // namespace Test
}  // namespace Cuda
}  // namespace Acts
