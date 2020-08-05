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

namespace {

/// Type of "other spacepoint" passed to the kernel
enum OtherSPType : int {
  BottomSP = 0, //< The "other" spacepoint is a bottom one
  TopSP = 1 //< The "other" spacepoint is a top one
};

} // private namespace

namespace Acts {
namespace Cuda {
namespace kernels {

template<int SPType>
__device__ float getDeltaR(float /*middleR*/, float /*otherR*/) {
  // This function should *never* be called.
  assert(false);
  return 0.0f;
}

template<>
__device__ float getDeltaR<BottomSP>(float middleR, float bottomR) {
  return middleR - bottomR;
}

template<>
__device__ float getDeltaR<TopSP>(float middleR, float topR) {
  return topR - middleR;
}

template<int SPType>
__device__ float getCotTheta(float /*middleZ*/, float /*otherZ*/,
                             float /*deltaR*/) {
  // This function should *never* be called.
  assert(false);
  return 0.0f;
}

template<>
__device__ float getCotTheta<BottomSP>(float middleZ, float bottomZ,
                                       float deltaR) {
  return (middleZ - bottomZ) / deltaR;
}

template<>
__device__ float getCotTheta<TopSP>(float middleZ, float topZ, float deltaR) {
  return (topZ - middleZ) / deltaR;
}

template<int SPType>
__global__ void findDublets(std::size_t nMiddleSP, const float* middleSPArray,
                            std::size_t nOtherSP, const float* otherSPArray,
                            float deltaRMin, float deltaRMax, float cotThetaMax,
                            float collisionRegionMin, float collisionRegionMax,
                            int* compCountArray, int* compArray) {

  // Figure out which dublet the kernel operates on.
  const std::size_t middleIndex = blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t otherIndex = blockIdx.y * blockDim.y + threadIdx.y;

  // If we're outside of bounds, stop here.
  if ((middleIndex >= nMiddleSP) || (otherIndex >= nOtherSP)) {
    return;
  }

  // Create helper objects on top of the arrays.
  const std::size_t middleSPSize[] = {nMiddleSP, details::SP_DIMENSIONS};
  DeviceMatrix<2, float> middleSPs(middleSPSize, middleSPArray);
  const std::size_t otherSPSize[] = {nOtherSP, details::SP_DIMENSIONS};
  DeviceMatrix<2, float> otherSPs(otherSPSize, otherSPArray);
  const std::size_t compSize[] = {nMiddleSP, nOtherSP};
  DeviceMatrix<2, int> compMatrix(compSize, compArray);

  std::size_t middleRIndex[] = {middleIndex, details::SP_R_INDEX};
  std::size_t middleZIndex[] = {middleIndex, details::SP_Z_INDEX};
  std::size_t otherRIndex[] = {otherIndex, details::SP_R_INDEX};
  std::size_t otherZIndex[] = {otherIndex, details::SP_Z_INDEX};

  // Access the parameters of interest for the two space points.
  const float middleR = middleSPs.get(middleRIndex);
  const float middleZ = middleSPs.get(middleZIndex);
  const float otherR = otherSPs.get(otherRIndex);
  const float otherZ = otherSPs.get(otherZIndex);

  // Calculate variables used in the compatibility check.
  const float deltaR = getDeltaR<SPType>(middleR, otherR);
  const float cotTheta = getCotTheta<SPType>(middleZ, otherZ, deltaR);
  const float zOrigin = middleZ - middleR * cotTheta;

  // Perform the compatibility check.
  const bool isCompatible = ((deltaR >= deltaRMin) && (deltaR <= deltaRMax) &&
                             (fabs(cotTheta) <= cotThetaMax) &&
                             (zOrigin >= collisionRegionMin) &&
                             (zOrigin <= collisionRegionMax));

  // If they are compatible, save their indices into the output matrix.
  if (isCompatible) {
    const int compRow = atomicAdd(compCountArray + middleIndex, 1);
    std::size_t compIndex[] = {middleIndex,
                               static_cast<std::size_t>(compRow)};
    compMatrix.set(compIndex, otherIndex);
  }
  return;
}

__global__ void countDublets(std::size_t nMiddleSP,
                             const int* middleBottomCountArray,
                             const int* middleTopCountArray,
                             details::DubletCounts* dubletCounts) {

  extern __shared__ details::DubletCounts sum[];

  // Get the thread identifier. Note that the kernel launch requests half as
  // many threads than how many elements we have in the arrays.
  const int middleIndex = blockIdx.x * blockDim.x * 2 + threadIdx.x;
  details::DubletCounts thisSum;
  thisSum.nDublets = ((middleIndex < nMiddleSP) ?
                      middleBottomCountArray[middleIndex] +
                      middleTopCountArray[middleIndex] : 0);
  thisSum.nTriplets = ((middleIndex < nMiddleSP) ?
                       middleBottomCountArray[middleIndex] *
                       middleTopCountArray[middleIndex] : 0);
  thisSum.maxMBDublets = ((middleIndex < nMiddleSP) ?
                          middleBottomCountArray[middleIndex] : 0);
  thisSum.maxMTDublets = ((middleIndex < nMiddleSP) ?
                          middleTopCountArray[middleIndex] : 0);
  if (middleIndex + blockDim.x < nMiddleSP) {
    thisSum.nDublets += (middleBottomCountArray[middleIndex + blockDim.x] +
                         middleTopCountArray[middleIndex + blockDim.x]);
    thisSum.nTriplets += (middleBottomCountArray[middleIndex + blockDim.x] *
                          middleTopCountArray[middleIndex + blockDim.x]);
    thisSum.maxMBDublets = max(middleBottomCountArray[middleIndex + blockDim.x],
                               thisSum.maxMBDublets);
    thisSum.maxMTDublets = max(middleTopCountArray[middleIndex + blockDim.x],
                               thisSum.maxMTDublets);
  }

  // Load the first sum step into shared memory.
  sum[threadIdx.x] = thisSum;
  __syncthreads();

  // Do the summation in some iterations.
  for (unsigned int i = blockDim.x / 2; i > 0; i>>=1) {
    if (threadIdx.x < i) {
      const details::DubletCounts& otherSum = sum[threadIdx.x + i];
      thisSum.nDublets += otherSum.nDublets;
      thisSum.nTriplets += otherSum.nTriplets;
      thisSum.maxMBDublets = max(thisSum.maxMBDublets, otherSum.maxMBDublets);
      thisSum.maxMTDublets = max(thisSum.maxMTDublets, otherSum.maxMTDublets);
      sum[threadIdx.x] = thisSum;
    }
    __syncthreads();
  }

  // Write the result of this execution block into the global memory.
  if (threadIdx.x == 0) {
    dubletCounts[blockIdx.x] = thisSum;
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
                 device_array<int>& middleBottomCountArray,
                 device_array<int>& middleBottomArray,
                 device_array<int>& middleTopCountArray,
                 device_array<int>& middleTopArray) {

  // Calculate the parallelisation for the middle<->bottom spacepoint
  // compatibility flagging.
  const dim3 blockSizeMB(1, maxBlockSize);
  const dim3 numBlocksMB((nMiddleSP + blockSizeMB.x - 1)/blockSizeMB.x,
                         (nBottomSP + blockSizeMB.y - 1)/blockSizeMB.y);

  // Launch the middle-bottom dublet finding.
  kernels::findDublets<BottomSP><<<numBlocksMB, blockSizeMB>>>(
      nMiddleSP, middleSPDeviceMatrix.get(),
      nBottomSP, bottomSPDeviceMatrix.get(),
      deltaRMin, deltaRMax, cotThetaMax, collisionRegionMin, collisionRegionMax,
      middleBottomCountArray.get(), middleBottomArray.get());
  ACTS_CUDA_ERROR_CHECK(cudaGetLastError());

  // Calculate the parallelisation for the middle<->top spacepoint
  // compatibility flagging.
  const dim3 blockSizeMT(1, maxBlockSize);
  const dim3 numBlocksMT((nMiddleSP + blockSizeMT.x - 1)/blockSizeMT.x,
                         (nTopSP    + blockSizeMT.y - 1)/blockSizeMT.y);

  // Launch the middle-bottom dublet finding.
  kernels::findDublets<TopSP><<<numBlocksMT, blockSizeMT>>>(
      nMiddleSP, middleSPDeviceMatrix.get(),
      nTopSP, topSPDeviceMatrix.get(),
      deltaRMin, deltaRMax, cotThetaMax, collisionRegionMin, collisionRegionMax,
      middleTopCountArray.get(), middleTopArray.get());
  ACTS_CUDA_ERROR_CHECK(cudaGetLastError());
  ACTS_CUDA_ERROR_CHECK(cudaDeviceSynchronize());
  return;
}

DubletCounts countDublets(std::size_t maxBlockSize, std::size_t nMiddleSP,
                          const device_array<int>& middleBottomCountArray,
                          const device_array<int>& middleTopCountArray) {

  // Calculate the parallelisation for the dublet counting.
  const int numBlocks = (nMiddleSP + maxBlockSize - 1) / maxBlockSize;
  const int sharedMem = maxBlockSize * sizeof(DubletCounts);

  // Create the small memory block in which we will get the count back for each
  // execution block.
  auto dubletCountsDevice = make_device_array<DubletCounts>(numBlocks);

  // Run the reduction kernel.
  kernels::countDublets<<<numBlocks, maxBlockSize, sharedMem>>>(
      nMiddleSP, middleBottomCountArray.get(), middleTopCountArray.get(),
      dubletCountsDevice.get());
  ACTS_CUDA_ERROR_CHECK(cudaGetLastError());
  ACTS_CUDA_ERROR_CHECK(cudaDeviceSynchronize());

  // Copy the sum(s) back to the host.
  auto dubletCountsHost = make_host_array<DubletCounts>(numBlocks);
  ACTS_CUDA_ERROR_CHECK(cudaMemcpy(dubletCountsHost.get(),
                                   dubletCountsDevice.get(),
                                   numBlocks * sizeof(DubletCounts),
                                   cudaMemcpyDeviceToHost));

  // Perform the final summation on the host. (Assuming that the number of
  // middle space points is not so large that it would make sense to do the
  // summation iteratively on the device.)
  DubletCounts result;
  for (int i = 0; i < numBlocks; ++i) {
    result.nDublets += dubletCountsHost.get()[i].nDublets;
    result.nTriplets += dubletCountsHost.get()[i].nTriplets;
    result.maxMBDublets = std::max(dubletCountsHost.get()[i].maxMBDublets,
                                   result.maxMBDublets);
    result.maxMTDublets = std::max(dubletCountsHost.get()[i].maxMTDublets,
                                   result.maxMTDublets);
  }
  return result;
}

}  // namespace details
}  // namespace Cuda
}  // namespace Acts
