// This file is part of the Acts project.
//
// Copyright (C) 2020 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

// CUDA plugin include(s).
#include "Acts/Plugins/Cuda/Seeding/FindDublets.hpp"
#include "Acts/Plugins/Cuda/Seeding/Types.hpp"
#include "Acts/Plugins/Cuda/Utilities/DeviceMatrix.cuh"
#include "Acts/Plugins/Cuda/Utilities/ErrorCheck.cuh"
#include "Acts/Plugins/Cuda/Utilities/HostMatrix.hpp"

// CUDA include(s).
#include <cuda_runtime.h>

// System include(s).
#include <cmath>

namespace Acts {
namespace Cuda {
namespace kernels {

__global__ void findDublets(std::size_t nInnerSP, const float* innerSPArray,
                            std::size_t nOuterSP, const float* outerSPArray,
                            float deltaRMin, float deltaRMax, float cotThetaMax,
                            float collisionRegionMin, float collisionRegionMax,
                            bool zOriginUseOuter, int* nCompPairs,
                            int* compPairArray) {

  // Figure out which dublet the kernel operates on.
  const std::size_t innerIndex = blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t outerIndex = blockIdx.y * blockDim.y + threadIdx.y;

  // If we're outside of bounds, stop here.
  if ((innerIndex >= nInnerSP) || (outerIndex >= nOuterSP)) {
    return;
  }

  // Create helper objects on top of the arrays.
  const std::size_t innerSPSize[] = {nInnerSP, details::SP_DIMENSIONS};
  DeviceMatrix<2, float> innerSP(innerSPSize, innerSPArray);
  const std::size_t outerSPSize[] = {nOuterSP, details::SP_DIMENSIONS};
  DeviceMatrix<2, float> outerSP(outerSPSize, outerSPArray);
  const std::size_t compPairSize[] = {nInnerSP * nOuterSP, 2};
  DeviceMatrix<2, int> compPairs(compPairSize, compPairArray);

  std::size_t innerRIndex[] = {innerIndex, details::SP_R_INDEX};
  std::size_t innerZIndex[] = {innerIndex, details::SP_Z_INDEX};
  std::size_t outerRIndex[] = {outerIndex, details::SP_R_INDEX};
  std::size_t outerZIndex[] = {outerIndex, details::SP_Z_INDEX};

  // Access the parameters of interest for the two space points.
  const float innerR = innerSP.get(innerRIndex);
  const float innerZ = innerSP.get(innerZIndex);
  const float outerR = outerSP.get(outerRIndex);
  const float outerZ = outerSP.get(outerZIndex);

  // Calculate variables used in the compatibility check.
  float deltaR = outerR - innerR;
  float cotTheta = (outerZ - innerZ) / deltaR;
  float zOrigin = (zOriginUseOuter ? outerZ - outerR * cotTheta :
                                     innerZ - innerR * cotTheta);

  // Perform the compatibility check.
  const bool isCompatible = ((deltaR >= deltaRMin) && (deltaR <= deltaRMax) &&
                             (fabs(cotTheta) <= cotThetaMax) &&
                             (zOrigin >= collisionRegionMin) &&
                             (zOrigin <= collisionRegionMax));

  // If they are compatible, save their indices into the output matrix.
  if (isCompatible) {
    const int compRow = atomicAdd(nCompPairs, 1);
    std::size_t compInnerIndex[] = {static_cast<std::size_t>(compRow), 0};
    std::size_t compOuterIndex[] = {static_cast<std::size_t>(compRow), 1};
    compPairs.set(compInnerIndex, innerIndex);
    compPairs.set(compOuterIndex, outerIndex);
  }
  return;
}

}  // namespace kernels

namespace details {

void findDublets(std::size_t maxBlockSize,
                 std::size_t nBottomSP,
                 const device_array<float>& bottomSPDeviceMatrix,
                 std::size_t nMiddleSP,
                 const device_array<float>& middleSPDeviceMatrix,
                 std::size_t nTopSP,
                 const device_array<float>& topSPDeviceMatrix,
                 float deltaRMin, float deltaRMax,
                 float cotThetaMax, float collisionRegionMin,
                 float collisionRegionMax,
                 ResultScalar<int>& nBottomMiddlePairs,
                 device_array<int>& bottomMiddlePairs,
                 ResultScalar<int>& nMiddleTopPairs,
                 device_array<int>& middleTopPairs) {

  // Calculate the parallelisation for the middle<->bottom spacepoint
  // compatibility flagging.
  const dim3 blockSizeBM(maxBlockSize, 1);
  const dim3 numBlocksBM((nBottomSP + blockSizeBM.x - 1)/blockSizeBM.x,
                         (nMiddleSP + blockSizeBM.y - 1)/blockSizeBM.y);

  // Variable helping with readability.
  static constexpr bool ZORIGIN_USE_OUTER = true;

  // Launch the middle-bottom dublet finding.
  kernels::findDublets<<<numBlocksBM, blockSizeBM>>>(
      nBottomSP, bottomSPDeviceMatrix.get(),
      nMiddleSP, middleSPDeviceMatrix.get(),
      deltaRMin, deltaRMax, cotThetaMax, collisionRegionMin, collisionRegionMax,
      ZORIGIN_USE_OUTER, nBottomMiddlePairs.getPtr(), bottomMiddlePairs.get());
  ACTS_CUDA_ERROR_CHECK(cudaGetLastError());

  // Calculate the parallelisation for the middle<->top spacepoint
  // compatibility flagging.
  const dim3 blockSizeMT(1, maxBlockSize);
  const dim3 numBlocksMT((nMiddleSP + blockSizeMT.x - 1)/blockSizeMT.x,
                         (nTopSP    + blockSizeMT.y - 1)/blockSizeMT.y);

  // Launch the middle-bottom dublet finding.
  kernels::findDublets<<<numBlocksMT, blockSizeMT>>>(
      nMiddleSP, middleSPDeviceMatrix.get(),
      nTopSP, topSPDeviceMatrix.get(),
      deltaRMin, deltaRMax, cotThetaMax, collisionRegionMin, collisionRegionMax,
      !ZORIGIN_USE_OUTER, nMiddleTopPairs.getPtr(), middleTopPairs.get());
  ACTS_CUDA_ERROR_CHECK(cudaGetLastError());
  ACTS_CUDA_ERROR_CHECK(cudaDeviceSynchronize());
  return;
}

}  // namespace details
}  // namespace Cuda
}  // namespace Acts
