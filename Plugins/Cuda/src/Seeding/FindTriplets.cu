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

__device__ void transformCoordinates(const details::SpacePoint& spM,
                                     const details::SpacePoint& sp,
                                     details::LinCircle& lc,
                                     bool bottom) {

  // Parameters of the middle spacepoint.
  float xM = spM.x;
  float yM = spM.y;
  float zM = spM.z;
  float rM = spM.radius;
  float varianceZM = spM.varianceZ;
  float varianceRM = spM.varianceR;
  float cosPhiM = xM / rM;
  float sinPhiM = yM / rM;

  // Parameters of the spacepoint being transformed.
  float deltaX = sp.x - xM;
  float deltaY = sp.y - yM;
  float deltaZ = sp.z - zM;
  // calculate projection fraction of spM->sp vector pointing in same
  // direction as
  // vector origin->spM (x) and projection fraction of spM->sp vector pointing
  // orthogonal to origin->spM (y)
  float x = deltaX * cosPhiM + deltaY * sinPhiM;
  float y = deltaY * cosPhiM - deltaX * sinPhiM;
  // 1/(length of M -> SP)
  float iDeltaR2 = 1. / (deltaX * deltaX + deltaY * deltaY);
  float iDeltaR = sqrtf(iDeltaR2);
  //
  int bottomFactor = 1 * (int(!bottom)) - 1 * (int(bottom));
  // cot_theta = (deltaZ/deltaR)
  float cot_theta = deltaZ * iDeltaR * bottomFactor;
  // VERY frequent (SP^3) access
  lc.cotTheta = cot_theta;
  // location on z-axis of this SP-duplet
  lc.Zo = zM - rM * cot_theta;
  lc.iDeltaR = iDeltaR;
  // transformation of circle equation (x,y) into linear equation (u,v)
  // x^2 + y^2 - 2x_0*x - 2y_0*y = 0
  // is transformed into
  // 1 - 2x_0*u - 2y_0*v = 0
  // using the following m_U and m_V
  // (u = A + B*v); A and B are created later on
  lc.U = x * iDeltaR2;
  lc.V = y * iDeltaR2;
  // error term for sp-pair without correlation of middle space point
  lc.Er = ((varianceZM + sp.varianceZ) +
           (cot_theta * cot_theta) * (varianceRM + sp.varianceR)) *
          iDeltaR2;
  return;
}

__global__ void transformCoordinates(int nDublets, int maxMBDublets,
                                int maxMTDublets,
                                std::size_t nBottomSP,
                                const details::SpacePoint* bottomSPArray,
                                std::size_t nMiddleSP,
                                const details::SpacePoint* middleSPArray,
                                std::size_t nTopSP,
                                const details::SpacePoint* topSPArray,
                                const int* middleBottomCountArray,
                                const int* middleBottomArray,
                                const int* middleTopCountArray,
                                const int* middleTopArray,
                                details::LinCircle* bottomSPLinTransArray,
                                details::LinCircle* topSPLinTransArray) {

  // Get the global index.
  const int dubletIndex = blockIdx.x * blockDim.x + threadIdx.x;

  // If we're out of bounds, finish right away.
  if (dubletIndex >= nDublets) {
    return;
  }

  // Create helper objects on top of the dublet matrices.
  const std::size_t middleBottomMatrixSize[] = {nMiddleSP, nBottomSP};
  DeviceMatrix<2, int> middleBottomMatrix(middleBottomMatrixSize,
                                          middleBottomArray);
  const std::size_t middleTopMatrixSize[] = {nMiddleSP, nTopSP};
  DeviceMatrix<2, int> middleTopMatrix(middleTopMatrixSize,
                                       middleTopArray);

  // Create helper objects on top of the LinCircle matrices.
  const std::size_t bottomSPLinTransMatrixSize[] =
      {nMiddleSP, static_cast<std::size_t>(maxMBDublets)};
  DeviceMatrix<2, details::LinCircle>
      bottomSPLinTransMatrix(bottomSPLinTransMatrixSize,
                             bottomSPLinTransArray);
  const std::size_t topSPLinTransMatrixSize[] =
      {nMiddleSP, static_cast<std::size_t>(maxMTDublets)};
  DeviceMatrix<2, details::LinCircle>
      topSPLinTransMatrix(topSPLinTransMatrixSize, topSPLinTransArray);

  // Find the dublet to transform.
  std::size_t middleIndex = 0;
  int runningIndex = dubletIndex;
  int tmpValue = 0;
  while (runningIndex >= (tmpValue = (middleBottomCountArray[middleIndex] +
                                      middleTopCountArray[middleIndex]))) {
    assert(middleIndex < nMiddleSP);
    middleIndex += 1;
    runningIndex -= tmpValue;
  }
  const bool transformBottom =
      ((runningIndex < middleBottomCountArray[middleIndex]) ? true : false);
  std::size_t bottomMatrixIndex = (transformBottom ? runningIndex : 0);
  std::size_t topMatrixIndex = (transformBottom ? 0 :
                                runningIndex -
                                middleBottomCountArray[middleIndex]);

  // Perform the transformation.
  if (transformBottom) {
    std::size_t middleBottomMatrixIndex[] = {middleIndex, bottomMatrixIndex};
    std::size_t bottomIndex = middleBottomMatrix.get(middleBottomMatrixIndex);
    assert(bottomIndex < nBottomSP);
    transformCoordinates(middleSPArray[middleIndex], bottomSPArray[bottomIndex],
                         bottomSPLinTransMatrix.getNC(middleBottomMatrixIndex),
                         true);
  } else {
    std::size_t middleTopMatrixIndex[] = {middleIndex, topMatrixIndex};
    std::size_t topIndex = middleTopMatrix.get(middleTopMatrixIndex);
    assert(topIndex < nTopSP);
    transformCoordinates(middleSPArray[middleIndex], topSPArray[topIndex],
                         topSPLinTransMatrix.getNC(middleTopMatrixIndex),
                         false);
  }

  return;
}

__global__ void findTriplets(int nTripletCandidates,
                             std::size_t nBottomSP,
                             const details::SpacePoint* bottomSPArray,
                             std::size_t nMiddleSP,
                             const details::SpacePoint* middleSPArray,
                             std::size_t nTopSP,
                             const details::SpacePoint* topSPArray,
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
  /*
  float xM = middleSPArray[middleIndex].x;
  float yM = middleSPArray[middleIndex].y;
  float zM = middleSPArray[middleIndex].z;
  float rM = middleSPArray[middleIndex].radius;
  float varianceZM = middleSPArray[middleIndex].varianceZ;
  float varianceRM = middleSPArray[middleIndex].varianceR;
  float cosPhiM = xM / rM;
  float sinPhiM = yM / rM;
  */

  return;
}

}  // namespace kernels

namespace details {

void findTriplets(int maxBlockSize, const DubletCounts& dubletCounts,
                  std::size_t nBottomSP,
                  const device_array<SpacePoint>& bottomSPArray,
                  std::size_t nMiddleSP,
                  const device_array<SpacePoint>& middleSPArray,
                  std::size_t nTopSP,
                  const device_array<SpacePoint>& topSPArray,
                  const device_array<int>& middleBottomCountArray,
                  const device_array<int>& middleBottomArray,
                  const device_array<int>& middleTopCountArray,
                  const device_array<int>& middleTopArray) {

  // Calculate the parallelisation for the parameter transformation.
  const int numBlocksLT =
      (dubletCounts.nDublets + maxBlockSize - 1) / maxBlockSize;

  // Create the arrays holding the linear transformed spacepoint parameters.
  auto bottomSPLinTransArray =
      make_device_array<LinCircle>(nMiddleSP * dubletCounts.maxMBDublets);
  auto topSPLinTransArray =
      make_device_array<LinCircle>(nMiddleSP * dubletCounts.maxMTDublets);

  // Launch the coordinate transformations.
  kernels::transformCoordinates<<<numBlocksLT, maxBlockSize>>>(
      dubletCounts.nDublets, dubletCounts.maxMBDublets,
      dubletCounts.maxMTDublets, nBottomSP, bottomSPArray.get(), nMiddleSP,
      middleSPArray.get(), nTopSP, topSPArray.get(),
      middleBottomCountArray.get(), middleBottomArray.get(),
      middleTopCountArray.get(), middleTopArray.get(),
      bottomSPLinTransArray.get(), topSPLinTransArray.get());
  ACTS_CUDA_ERROR_CHECK(cudaGetLastError());
  ACTS_CUDA_ERROR_CHECK(cudaDeviceSynchronize());

  // Calculate the parallelisation for the triplet finding.
  const int numBlocksFT =
      (dubletCounts.nTriplets + maxBlockSize - 1) / maxBlockSize;

  // Launch the triplet finding.
  kernels::findTriplets<<<numBlocksFT, maxBlockSize>>>(
      dubletCounts.nTriplets, nBottomSP, bottomSPArray.get(), nMiddleSP,
      middleSPArray.get(), nTopSP, topSPArray.get(),
      middleBottomCountArray.get(), middleBottomArray.get(),
      middleTopCountArray.get(), middleTopArray.get());
  ACTS_CUDA_ERROR_CHECK(cudaGetLastError());
  ACTS_CUDA_ERROR_CHECK(cudaDeviceSynchronize());
  return;
}

}  // namespace details
}  // namespace Cuda
}  // namespace Acts
