// This file is part of the Acts project.
//
// Copyright (C) 2020 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

// CUDA plugin include(s).
#include "Acts/Plugins/Cuda/Seeding/FindTriplets.hpp"
#include "Acts/Plugins/Cuda/Seeding/Types.hpp"
#include "Acts/Plugins/Cuda/Utilities/DeviceMatrix.cuh"
#include "Acts/Plugins/Cuda/Utilities/ErrorCheck.cuh"
#include "Acts/Plugins/Cuda/Utilities/HostMatrix.hpp"

// CUDA include(s).
#include <cuda_runtime.h>

// System include(s).
#include <cmath>

#include <iostream>

namespace Acts {
namespace Cuda {
namespace kernels {

__global__ void findTriplets(int nTripletCandidates,
                             std::size_t nBottomSP, const float* bottomSPArray,
                             std::size_t nMiddleSP, const float* middleSPArray,
                             std::size_t nTopSP, const float* topSPArray,
                             const int* middleBottomCountArray,
                             const int* middleBottomArray,
                             const int* middleTopCountArray,
                             const int* middleTopArray) {

  // Get the global index.
  const int tripletIndex = blockIdx.x * blockDim.x + threadIdx.x;

  // If we're out of bounds, finish right away.
  if (tripletIndex >= nTripletCandidates) {
    return;
  }

  // Create helper objects on top of the spacepoint arrays.
  const std::size_t bottomSPSize[] = {nBottomSP, details::SP_DIMENSIONS};
  DeviceMatrix<2, float> bottomSPs(bottomSPSize, bottomSPArray);
  const std::size_t middleSPSize[] = {nMiddleSP, details::SP_DIMENSIONS};
  DeviceMatrix<2, float> middleSPs(middleSPSize, middleSPArray);
  const std::size_t topSPSize[] = {nTopSP, details::SP_DIMENSIONS};
  DeviceMatrix<2, float> topSPs(topSPSize, topSPArray);

  // Create helper objects on top of the dublet matrices.
  const std::size_t middleBottomMatrixSize[] = {nMiddleSP, nBottomSP};
  DeviceMatrix<2, int> middleBottomMatrix(middleBottomMatrixSize,
                                          middleBottomArray);
  const std::size_t middleTopMatrixSize[] = {nMiddleSP, nTopSP};
  DeviceMatrix<2, int> middleTopMatrix(middleTopMatrixSize,
                                       middleTopArray);

  // Find the dublet pair to evaluate.
  std::size_t middleIndex = 0;
  int runningIndex = tripletIndex;
  int tmpValue = 0;
  while (runningIndex >= (tmpValue = (middleBottomCountArray[middleIndex] *
                                      middleTopCountArray[middleIndex]))) {
    assert(middleIndex < nMiddleSP);
    middleIndex += 1;
    runningIndex -= tmpValue;
  }
  std::size_t bottomMatrixIndex =
    runningIndex / middleTopCountArray[middleIndex];
  assert(bottomMatrixIndex < middleBottomCountArray[middleIndex]);
  std::size_t topMatrixIndex = runningIndex % middleTopCountArray[middleIndex];
  std::size_t middleBottomMatrixIndex[] = {middleIndex, bottomMatrixIndex};
  std::size_t middleTopMatrixIndex[] = {middleIndex, topMatrixIndex};
  std::size_t bottomIndex = middleBottomMatrix.get(middleBottomMatrixIndex);
  assert(bottomIndex < nBottomSP);
  std::size_t topIndex = middleTopMatrix.get(middleTopMatrixIndex);
  assert(topIndex < nTopSP);

  // Extract the properties of the selected spacepoints.
  std::size_t middleXIndex[] = {middleIndex, details::SP_X_INDEX};
  std::size_t middleYIndex[] = {middleIndex, details::SP_Y_INDEX};
  std::size_t middleZIndex[] = {middleIndex, details::SP_Z_INDEX};
  std::size_t middleRIndex[] = {middleIndex, details::SP_R_INDEX};
  std::size_t middleVZIndex[] = {middleIndex, details::SP_VZ_INDEX};
  std::size_t middleVRIndex[] = {middleIndex, details::SP_VR_INDEX};

  float xM = middleSPs.get(middleXIndex);
  float yM = middleSPs.get(middleYIndex);
  float zM = middleSPs.get(middleZIndex);
  float rM = middleSPs.get(middleRIndex);
  float varianceZM = middleSPs.get(middleVZIndex);
  float varianceRM = middleSPs.get(middleVRIndex);
  float cosPhiM = xM / rM;
  float sinPhiM = yM / rM;

  return;
}

}  // namespace kernels

namespace details {

void findTriplets(int maxBlockSize, const DubletCounts& dubletCounts,
                  std::size_t nBottomSP,
                  const device_array<float>& bottomSPDeviceMatrix,
                  std::size_t nMiddleSP,
                  const device_array<float>& middleSPDeviceMatrix,
                  std::size_t nTopSP,
                  const device_array<float>& topSPDeviceMatrix,
                  const device_array<int>& middleBottomCountArray,
                  const device_array<int>& middleBottomArray,
                  const device_array<int>& middleTopCountArray,
                  const device_array<int>& middleTopArray) {

  // Calculate the parallelisation for the triplet finding.
  const int numBlocks =
      (dubletCounts.nTriplets + maxBlockSize - 1) / maxBlockSize;

  // Launch the triplet finding.
  kernels::findTriplets<<<numBlocks, maxBlockSize>>>(
      dubletCounts.nTriplets, nBottomSP, bottomSPDeviceMatrix.get(), nMiddleSP,
      middleSPDeviceMatrix.get(), nTopSP, topSPDeviceMatrix.get(),
      middleBottomCountArray.get(), middleBottomArray.get(),
      middleTopCountArray.get(), middleTopArray.get());
  ACTS_CUDA_ERROR_CHECK(cudaGetLastError());
  ACTS_CUDA_ERROR_CHECK(cudaDeviceSynchronize());
  return;
}

}  // namespace details
}  // namespace Cuda
}  // namespace Acts
