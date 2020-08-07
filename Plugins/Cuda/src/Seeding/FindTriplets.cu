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
#include "Acts/Plugins/Cuda/Utilities/ErrorCheck.cuh"
#include "Acts/Plugins/Cuda/Utilities/MatrixMacros.hpp"

// CUDA include(s).
#include <cuda_runtime.h>

// System include(s).
#include <cassert>
#include <cmath>
#include <cstring>

#include <iostream>

namespace Acts {
namespace Cuda {
namespace kernels {

/// Function performing coordinate transformation for one spacepoint pair
///
/// @param spM    The middle spacepoint to use
/// @param sp     The "other" spacepoint to use
/// @param bottom @c true If the "other" spacepoint is a bottom one, @c false
///               otherwise
__device__ details::LinCircle transformCoordinates(
    const details::SpacePoint& spM, const details::SpacePoint& sp,
    bool bottom) {

  // Create the result object.
  details::LinCircle result;

  // Parameters of the middle spacepoint.
  const float cosPhiM = spM.x / spM.radius;
  const float sinPhiM = spM.y / spM.radius;

  // (Relative) Parameters of the spacepoint being transformed.
  const float deltaX = sp.x - spM.x;
  const float deltaY = sp.y - spM.y;
  const float deltaZ = sp.z - spM.z;

  // calculate projection fraction of spM->sp vector pointing in same
  // direction as
  // vector origin->spM (x) and projection fraction of spM->sp vector pointing
  // orthogonal to origin->spM (y)
  const float x = deltaX * cosPhiM + deltaY * sinPhiM;
  const float y = deltaY * cosPhiM - deltaX * sinPhiM;
  // 1/(length of M -> SP)
  const float iDeltaR2 = 1. / (deltaX * deltaX + deltaY * deltaY);
  const float iDeltaR = sqrtf(iDeltaR2);
  //
  const int bottomFactor = 1 * (int(!bottom)) - 1 * (int(bottom));
  // cot_theta = (deltaZ/deltaR)
  const float cot_theta = deltaZ * iDeltaR * bottomFactor;
  // VERY frequent (SP^3) access
  result.cotTheta = cot_theta;
  // location on z-axis of this SP-duplet
  result.Zo = spM.z - spM.radius * cot_theta;
  result.iDeltaR = iDeltaR;
  // transformation of circle equation (x,y) into linear equation (u,v)
  // x^2 + y^2 - 2x_0*x - 2y_0*y = 0
  // is transformed into
  // 1 - 2x_0*u - 2y_0*v = 0
  // using the following m_U and m_V
  // (u = A + B*v); A and B are created later on
  result.U = x * iDeltaR2;
  result.V = y * iDeltaR2;
  // error term for sp-pair without correlation of middle space point
  result.Er = ((spM.varianceZ + sp.varianceZ) +
               (cot_theta * cot_theta) * (spM.varianceR + sp.varianceR)) *
              iDeltaR2;
  return result;
}

/// Kernel performing coordinate transformation on all created dublets
///
/// @param nDublets The total number of dublets found
/// @param maxMBDublets The maximal middle-bottom dublets found for any middle
///                     spacepoint
/// @param maxMTDublets The maximal middle-top dublets found for any middle
///                     spacepoint
/// @param nBottomSP The total number of bottom spacepoints
/// @param bottomSPArray 1-dimensional array to all bottom spacepoints
/// @param nMiddleSP The total number of middle spacepoints
/// @param middleSPArray 1-dimensional array to all middle spacepoints
/// @param nTopSP The total number of top spacepoints
/// @param topSPArray 1-dimensional array to all the top spacepoints
/// @param middleBottomCountArray 1-dimensional array of the middle-bottom
///                               dublet counts
/// @param middleBottomArray 2-dimensional matrix with the bottom spacepoint
///                          indices assigned to a given middle spacepoint
/// @param middleTopCountArray 1-dimensional array of the middle-top dublet
///                            counts
/// @param middleTopArray 2-dimensional matrix with the top spacepoint indices
///                       assigned to a given middle spacepoint
/// @param bottomSPLinTransArray 2-dimensional matrix indexed the same way as
///                              @c middleBottomArray
/// @param topSPLinTransArray 2-dimensional matrix indexed the same way as
///                           @c middleTopArray
///
__global__ void transformCoordinates(
    int nDublets, int maxMBDublets, int maxMTDublets,
    std::size_t nBottomSP, const details::SpacePoint* bottomSPArray,
    std::size_t nMiddleSP, const details::SpacePoint* middleSPArray,
    std::size_t nTopSP, const details::SpacePoint* topSPArray,
    const int* middleBottomCountArray, const int* middleBottomArray,
    const int* middleTopCountArray, const int* middleTopArray,
    details::LinCircle* bottomSPLinTransArray,
    details::LinCircle* topSPLinTransArray) {

  // Get the global index.
  const int dubletIndex = blockIdx.x * blockDim.x + threadIdx.x;

  // If we're out of bounds, finish right away.
  if (dubletIndex >= nDublets) {
    return;
  }

  // Find the dublet to transform.
  std::size_t middleIndex = 0;
  int runningIndex = dubletIndex;
  int tmpValue = 0;
  while (runningIndex >= (tmpValue = (middleBottomCountArray[middleIndex] +
                                      middleTopCountArray[middleIndex]))) {
    middleIndex += 1;
    assert(middleIndex < nMiddleSP);
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
    std::size_t bottomIndex =
        ACTS_CUDA_MATRIX2D_ELEMENT(middleBottomArray, nMiddleSP, nBottomSP,
                                   middleIndex, bottomMatrixIndex);
    assert(bottomIndex < nBottomSP);
    ACTS_CUDA_MATRIX2D_ELEMENT(
      bottomSPLinTransArray, nMiddleSP, maxMBDublets, middleIndex,
      bottomMatrixIndex) =
        transformCoordinates(middleSPArray[middleIndex],
                             bottomSPArray[bottomIndex], true);
  } else {
    std::size_t topIndex =
        ACTS_CUDA_MATRIX2D_ELEMENT(middleTopArray, nMiddleSP, nTopSP,
                                   middleIndex, topMatrixIndex);
    assert(topIndex < nTopSP);
    ACTS_CUDA_MATRIX2D_ELEMENT(
      topSPLinTransArray, nMiddleSP, maxMTDublets, middleIndex,
      topMatrixIndex) =
        transformCoordinates(middleSPArray[middleIndex], topSPArray[topIndex],
                             false);
  }

  return;
}

/// Kernel used for finding all the triplet candidates
///
/// @param middleIndex The middle spacepoint index to run the triplet search for
/// @param maxMBDublets The maximal middle-bottom dublets found for any middle
///                     spacepoint
/// @param maxMTDublets The maximal middle-top dublets found for any middle
///                     spacepoint
/// @param maxTriplets The maximum number of triplets for which memory is booked
/// @param nBottomSP The total number of bottom spacepoints
/// @param bottomSPArray 1-dimensional array to all bottom spacepoints
/// @param nMiddleSP The total number of middle spacepoints
/// @param middleSPArray 1-dimensional array to all middle spacepoints
/// @param nTopSP The total number of top spacepoints
/// @param topSPArray 1-dimensional array to all the top spacepoints
/// @param middleBottomCountArray 1-dimensional array of the middle-bottom
///                               dublet counts
/// @param middleBottomArray 2-dimensional matrix with the bottom spacepoint
///                          indices assigned to a given middle spacepoint
/// @param middleTopCountArray 1-dimensional array of the middle-top dublet
///                            counts
/// @param middleTopArray 2-dimensional matrix with the top spacepoint indices
///                       assigned to a given middle spacepoint
/// @param bottomSPLinTransArray 2-dimensional matrix indexed the same way as
///                              @c middleBottomArray
/// @param topSPLinTransArray 2-dimensional matrix indexed the same way as
///                           @c middleTopArray
/// @param maxScatteringAngle2 Parameter from @c Acts::SeedfinderConfig
/// @param sigmaScattering Parameter from @c Acts::SeedfinderConfig
/// @param minHelixDiameter2 Parameter from @c Acts::SeedfinderConfig
/// @param pT2perRadius Parameter from @c Acts::SeedfinderConfig
/// @param impactMax Parameter from @c Acts::SeedfinderConfig
/// @param impactWeightFactor Parameter from @c Acts::SeedfinderConfig
/// @param tripletsPerBottomDublet 1-dimensional array of the triplet counts for
///                                each bottom spacepoint
/// @param tripletIndices 2-dimensional matrix of the indices of the triplets
///                       created for each middle-bottom spacepoint dublet
/// @param maxTripletsPerSpB Pointer to the scalar outputting the maximum number
///                          of triplets found for any bottom spacepoint dublet
/// @param tripletCount Pointer to the scalar counting the total number of
///                     triplets created by the kernel
/// @param tripletArray 1-dimensional array of all reconstructed triplet
///                     candidates
///
__global__ void findTriplets(
    std::size_t middleIndex, int maxMBDublets, int maxMTDublets,
    int maxTriplets,
    std::size_t nBottomSP, const details::SpacePoint* bottomSPArray,
    std::size_t nMiddleSP, const details::SpacePoint* middleSPArray,
    std::size_t nTopSP, const details::SpacePoint* topSPArray,
    const int* middleBottomCountArray, const int* middleBottomArray,
    const int* middleTopCountArray, const int* middleTopArray,
    const details::LinCircle* bottomSPLinTransArray,
    const details::LinCircle* topSPLinTransArray,
    float maxScatteringAngle2, float sigmaScattering, float minHelixDiameter2,
    float pT2perRadius, float impactMax, float impactWeightFactor,
    int* tripletsPerBottomDublet, std::size_t* tripletIndices,
    int* maxTripletsPerSpB, int* tripletCount, details::Triplet* tripletArray) {

  // A sanity check.
  assert(middleIndex < nMiddleSP);

  // The total number of dublets for this middle spacepoint.
  const int middleBottomDublets = middleBottomCountArray[middleIndex];
  const int middleTopDublets = middleTopCountArray[middleIndex];

  // Get the indices of the dublets to operate on.
  const std::size_t bottomDubletIndex = blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t topDubletIndex = blockIdx.y * blockDim.y + threadIdx.y;
  if ((bottomDubletIndex >= middleBottomDublets) ||
      (topDubletIndex >= middleTopDublets)) {
    return;
  }

  // Get the indices of the spacepoints to operate on.
  const std::size_t bottomIndex = middleBottomArray[bottomDubletIndex];
  assert(bottomIndex < nBottomSP);
  const std::size_t topIndex = middleTopArray[topDubletIndex];
  assert(topIndex < nTopSP);

  // Load the transformed coordinates of the bottom spacepoint into the thread.
  const details::LinCircle lb =
      ACTS_CUDA_MATRIX2D_ELEMENT(bottomSPLinTransArray, nMiddleSP, maxMBDublets,
                                 middleIndex, bottomDubletIndex);

  // 1+(cot^2(theta)) = 1/sin^2(theta)
  float iSinTheta2 = (1. + lb.cotTheta * lb.cotTheta);
  // calculate max scattering for min momentum at the seed's theta angle
  // scaling scatteringAngle^2 by sin^2(theta) to convert pT^2 to p^2
  // accurate would be taking 1/atan(thetaBottom)-1/atan(thetaTop) <
  // scattering
  // but to avoid trig functions we approximate cot by scaling by
  // 1/sin^4(theta)
  // resolving with pT to p scaling --> only divide by sin^2(theta)
  // max approximation error for allowed scattering angles of 0.04 rad at
  // eta=infinity: ~8.5%
  float scatteringInRegion2 = maxScatteringAngle2 * iSinTheta2;
  // multiply the squared sigma onto the squared scattering
  scatteringInRegion2 *= sigmaScattering * sigmaScattering;

  // Load the transformed coordinates of the top spacepoint into the thread.
  const details::LinCircle lt =
      ACTS_CUDA_MATRIX2D_ELEMENT(topSPLinTransArray, nMiddleSP, maxMTDublets,
                                 middleIndex, topDubletIndex);

  // Load the parameters of the middle spacepoint into the thread.
  const details::SpacePoint spM = middleSPArray[middleIndex];

  // add errors of spB-spM and spM-spT pairs and add the correlation term
  // for errors on spM
  float error2 = lt.Er + lb.Er +
                 2 * (lb.cotTheta * lt.cotTheta * spM.varianceR +
                      spM.varianceZ) * lb.iDeltaR * lt.iDeltaR;

  float deltaCotTheta = lb.cotTheta - lt.cotTheta;
  float deltaCotTheta2 = deltaCotTheta * deltaCotTheta;
  float dCotThetaMinusError2 = 0.0f;

  // if the error is larger than the difference in theta, no need to
  // compare with scattering
  if (deltaCotTheta2 - error2 > 0) {
    deltaCotTheta = fabs(deltaCotTheta);
    // if deltaTheta larger than the scattering for the lower pT cut, skip
    float error = sqrtf(error2);
    dCotThetaMinusError2 =
        deltaCotTheta2 + error2 - 2 * deltaCotTheta * error;
    // avoid taking root of scatteringInRegion
    // if left side of ">" is positive, both sides of unequality can be
    // squared
    // (scattering is always positive)
    if (dCotThetaMinusError2 > scatteringInRegion2) {
      return;
    }
  }

  // protects against division by 0
  float dU = lt.U - lb.U;
  if (dU == 0.) {
    return;
  }
  // A and B are evaluated as a function of the circumference parameters
  // x_0 and y_0
  float A = (lt.V - lb.V) / dU;
  float S2 = 1. + A * A;
  float B = lb.V - A * lb.U;
  float B2 = B * B;
  // sqrt(S2)/B = 2 * helixradius
  // calculated radius must not be smaller than minimum radius
  if (S2 < B2 * minHelixDiameter2) {
    return;
  }
  // 1/helixradius: (B/sqrt(S2))/2 (we leave everything squared)
  float iHelixDiameter2 = B2 / S2;
  // calculate scattering for p(T) calculated from seed curvature
  float pT2scatter = 4 * iHelixDiameter2 * pT2perRadius;
  // TODO: include upper pT limit for scatter calc
  // convert p(T) to p scaling by sin^2(theta) AND scale by 1/sin^4(theta)
  // from rad to deltaCotTheta
  float p2scatter = pT2scatter * iSinTheta2;
  // if deltaTheta larger than allowed scattering for calculated pT, skip
  if ((deltaCotTheta2 - error2 > 0) &&
      (dCotThetaMinusError2 >
       p2scatter * sigmaScattering * sigmaScattering)) {
    return;
  }
  // A and B allow calculation of impact params in U/V plane with linear
  // function
  // (in contrast to having to solve a quadratic function in x/y plane)
  float Im = fabs((A - B * spM.radius) * spM.radius);

  // Check if the triplet candidate should be accepted.
  if (Im > impactMax) {
    return;
  }

  // Reserve elements (positions) in the global matrices/arrays.
  int tripletIndexRow =
    atomicAdd(tripletsPerBottomDublet + bottomDubletIndex, 1);
  assert(tripletIndexRow < maxMTDublets);
  int tripletIndex = atomicAdd(tripletCount, 1);
  assert(tripletIndex < maxTriplets);

  // Collect the maximal value of tripletIndexRow + 1 (since we want the
  // count, not the index values) for the next kernel.
  atomicMax(maxTripletsPerSpB, tripletIndexRow + 1);

  // Save the index of the triplet candidate, which will be created now.
  ACTS_CUDA_MATRIX2D_ELEMENT(
    tripletIndices, maxMBDublets, maxMTDublets, bottomDubletIndex,
    tripletIndexRow) = tripletIndex;

  // Now store the triplet in the above mentioned location.
  details::Triplet triplet = {bottomIndex, topIndex, Im,
                              B / sqrtf(S2), -(Im * impactWeightFactor)};
  tripletArray[tripletIndex] = triplet;

  return;
}

/// Code copied from @c Acts::ATLASCuts
__device__ float seedWeight(const details::SpacePoint& bottom,
                            const details::SpacePoint&,
                            const details::SpacePoint& top) {

  float weight = 0;
  if (bottom.radius > 150) {
    weight = 400;
  }
  if (top.radius < 150) {
    weight = 200;
  }
  return weight;
}

/// Code copied from @c Acts::ATLASCuts
__device__ bool singleSeedCut(float weight, const details::SpacePoint& bottom,
                              const details::SpacePoint&,
                              const details::SpacePoint&) {

  return !(bottom.radius > 150. && weight < 380.);
}

/// Kernel performing the "2 fixed spacepoint filtering" of the triplets
///
/// @param middleIndex The middle spacepoint index to run the triplet filter for
/// @param maxMBDublets The maximal middle-bottom dublets found for any middle
///                     spacepoint
/// @param maxMTDublets The maximal middle-top dublets found for any middle
///                     spacepoint
/// @param middleBottomDublets The total number of middle-bottom spacepoint
///                            dublets for this middle spacepoint
/// @param nBottomSP The total number of bottom spacepoints
/// @param bottomSPArray 1-dimensional array to all bottom spacepoints
/// @param nMiddleSP The total number of middle spacepoints
/// @param middleSPArray 1-dimensional array to all middle spacepoints
/// @param nTopSP The total number of top spacepoints
/// @param topSPArray 1-dimensional array to all the top spacepoints
/// @param tripletsPerBottomDublet 1-dimensional array of the number of triplets
///                                found for every middle-bottom spacepoint
///                                dublet
/// @param tripletIndices 2-dimensional matrix of the indices of the triplets
///                       created for each middle-bottom spacepoint dublet
/// @param nAllTriplets Pointer to the scalar number of triplets found in total
/// @param allTriplets 1-dimensional array of all the found triplets
/// @param deltaInvHelixDiameter Parameter from @c Acts::Cuda::SeedFilterConfig
/// @param deltaRMin Parameter from @c Acts::Cuda::SeedFilterConfig
/// @param compatSeedWeight Parameter from @c Acts::Cuda::SeedFilterConfig
/// @param compatSeedLimit Parameter from @c Acts::Cuda::SeedFilterConfig
/// @param nFilteredTriplets Pointer to the scalar counting all triplets that
///                          survive this filter
/// @param filteredTriplets 1-dimensional array of triplets that survive this
///                         filter
///
__global__ void filterTriplets2Sp(
    std::size_t middleIndex, int maxMBDublets, int maxMTDublets,
    int middleBottomDublets,
    std::size_t nBottomSP, const details::SpacePoint* bottomSPArray,
    std::size_t nMiddleSP, const details::SpacePoint* middleSPArray,
    std::size_t nTopSP, const details::SpacePoint* topSPArray,
    const int* tripletsPerBottomDublet, const std::size_t* tripletIndices,
    const int* nAllTriplets, const details::Triplet* allTriplets,
    float deltaInvHelixDiameter, float deltaRMin, float compatSeedWeight,
    std::size_t compatSeedLimit,
    int* nFilteredTriplets, details::Triplet* filteredTriplets) {

  // A sanity check.
  assert(middleIndex < nMiddleSP);

  // Get the indices of the objects to operate on.
  const std::size_t bottomDubletIndex = blockIdx.x * blockDim.x + threadIdx.x;
  if (bottomDubletIndex >= middleBottomDublets) {
    return;
  }
  const std::size_t nTriplets = tripletsPerBottomDublet[bottomDubletIndex];
  const std::size_t tripletMatrixIndex = blockIdx.y * blockDim.y + threadIdx.y;
  if (tripletMatrixIndex >= nTriplets) {
    return;
  }

  // Get the index of this triplet.
  const int triplet1Index =
      ACTS_CUDA_MATRIX2D_ELEMENT(tripletIndices, maxMBDublets, maxMTDublets,
                                 bottomDubletIndex, tripletMatrixIndex);
  assert(triplet1Index < *nAllTriplets);

  // Load this triplet into the thread.
  details::Triplet triplet1 = allTriplets[triplet1Index];

  // Pre-compute some variables.
  float lowerLimitCurv = triplet1.invHelixDiameter - deltaInvHelixDiameter;
  float upperLimitCurv = triplet1.invHelixDiameter + deltaInvHelixDiameter;
  float currentTop_r = topSPArray[triplet1.topIndex].radius;

  // Allow only a maximum number of top spacepoints in the filtering. Since a
  // limit is coming from @c compatSeedLimit anyway, this could potentially be
  // re-written with an array alocation, instead of statically defining the
  // array's size.
  static constexpr std::size_t MAX_TOP_SP = 10;
  assert(compatSeedLimit < MAX_TOP_SP);
  float compatibleSeedR[MAX_TOP_SP];
  std::size_t nCompatibleSeedR = 0;

  // Loop over all the other triplets found for this bottom-middle dublet.
  for (std::size_t i = 0; i < nTriplets; ++i) {

    // Don't consider the same triplet that the thread is evaluating in the
    // first place.
    if (i == tripletMatrixIndex) {
      continue;
    }
    // Get the index of the second triplet.
    const int triplet2Index =
        ACTS_CUDA_MATRIX2D_ELEMENT(tripletIndices, maxMBDublets, maxMTDublets,
                                   bottomDubletIndex, i);
    assert(triplet2Index < *nAllTriplets);
    assert(triplet2Index != triplet1Index);

    // Load the second triplet into the thread.
    const details::Triplet triplet2 = allTriplets[triplet2Index];
    assert(triplet1.bottomIndex == triplet2.bottomIndex);

    // compared top SP should have at least deltaRMin distance
    float otherTop_r = topSPArray[triplet2.topIndex].radius;
    float deltaR = currentTop_r - otherTop_r;
    if (fabs(deltaR) < deltaRMin) {
      continue;
    }

    // curvature difference within limits?
    // TODO: how much slower than sorting all vectors by curvature
    // and breaking out of loop? i.e. is vector size large (e.g. in jets?)
    if (triplet2.invHelixDiameter < lowerLimitCurv) {
      continue;
    }
    if (triplet2.invHelixDiameter > upperLimitCurv) {
      continue;
    }

    bool newCompSeed = true;
    for (std::size_t k = 0; k < nCompatibleSeedR; ++k) {
      // original ATLAS code uses higher min distance for 2nd found compatible
      // seed (20mm instead of 5mm)
      // add new compatible seed only if distance larger than rmin to all
      // other compatible seeds
      if (fabs(compatibleSeedR[k] - otherTop_r) < deltaRMin) {
        newCompSeed = false;
        break;
      }
    }
    if (newCompSeed) {
      compatibleSeedR[nCompatibleSeedR++] = otherTop_r;
      assert(nCompatibleSeedR < MAX_TOP_SP);
      triplet1.weight += compatSeedWeight;
    }
    if (nCompatibleSeedR >= compatSeedLimit) {
      break;
    }
  }

  // Decide whether to keep the triplet or not.
  triplet1.weight += seedWeight(bottomSPArray[triplet1.bottomIndex],
                                middleSPArray[middleIndex],
                                topSPArray[triplet1.topIndex]);
  if (!singleSeedCut(triplet1.weight, bottomSPArray[triplet1.bottomIndex],
                     middleSPArray[middleIndex],
                     topSPArray[triplet1.topIndex])) {
    return;
  }

  // Put the triplet into the "filtered list".
  const int tripletRow = atomicAdd(nFilteredTriplets, 1);
  filteredTriplets[tripletRow] = triplet1;
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
                  const device_array<int>& middleTopArray,
                  float maxScatteringAngle2, float sigmaScattering,
                  float minHelixDiameter2, float pT2perRadius,
                  float impactMax, float impactWeightFactor,
                  float deltaInvHelixDiameter, float deltaRMin,
                  float compatSeedWeight, std::size_t compatSeedLimit) {

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

  // Copy the dublet counts back to the host.
  auto middleBottomCountsHost = make_host_array<int>(nMiddleSP);
  copyToHost(middleBottomCountsHost, middleBottomCountArray, nMiddleSP);
  auto middleTopCountsHost = make_host_array<int>(nMiddleSP);
  copyToHost(middleTopCountsHost, middleTopCountArray, nMiddleSP);

  // The maximal number of triplets we should consider per middle spacepoint.
  const int maxTriplets = dubletCounts.maxMBDublets * dubletCounts.maxMTDublets;

  // Helper variables for handling the various object counts in device memory.
  enum ObjectCountType : int {
    AllTriplets = 0, ///< All viable triplets
    FilteredTriplets = 1, ///< Triplets after the "2SpFixed" filtering
    FinalTriplets = 2, ///< Triplets after the "1SpFixed" filtering
    MaxTripletsPerSpB = 3, ///< Maximal number of triplets found per SpB
    NObjectCountTypes = 4 ///< The number of different object/counter types
  };

  // Set up the object counters in device memory. The host array is only used to
  // reset the device memory before every iteration.
  auto objectCountsHostNull = make_host_array<int>(NObjectCountTypes);
  memset(objectCountsHostNull.get(), 0, NObjectCountTypes * sizeof(int));
  auto objectCountsDevice = make_device_array<int>(NObjectCountTypes);

  // Allocate enough memory for triplet candidates that would suffice for every
  // middle spacepoint.
  auto allTripletsArray = make_device_array<Triplet>(maxTriplets);
  auto filteredTripletsArray = make_device_array<Triplet>(maxTriplets);
  auto finalTripletsArray = make_device_array<Triplet>(maxTriplets);

  // Allocate and initialise the array holding the per bottom dublet triplet
  // numbers.
  auto tripletsPerBottomDubletHost =
      make_host_array<int>(dubletCounts.maxMBDublets);
  memset(tripletsPerBottomDubletHost.get(), 0,
         dubletCounts.maxMBDublets * sizeof(int));
  auto tripletsPerBottomDubletDevice =
      make_device_array<int>(dubletCounts.maxMBDublets);

  // Allocate the array holding the indices of the triplets found for a given
  // bottom-middle spacepoint combination.
  auto tripletIndices =
      make_device_array<std::size_t>(dubletCounts.maxMBDublets *
                                     dubletCounts.maxMTDublets);

  int allTriplets = 0, filteredTriplets = 0;
  auto objectCountsHost = make_host_array<int>(NObjectCountTypes);

  // Execute the triplet finding and filtering separately for each middle
  // spacepoint.
  for (std::size_t middleIndex = 0; middleIndex < nMiddleSP; ++middleIndex) {

    // The number of bottom-middle and middle-top dublets found for this middle
    // spacepoint.
    const int middleBottomDublets = middleBottomCountsHost.get()[middleIndex];
    const int middleTopDublets = middleTopCountsHost.get()[middleIndex];
    if ((middleBottomDublets == 0) || (middleTopDublets == 0)) {
      continue;
    }

    // Reset device arrays.
    copyToDevice(objectCountsDevice, objectCountsHostNull, NObjectCountTypes);
    copyToDevice(tripletsPerBottomDubletDevice, tripletsPerBottomDubletHost,
                 dubletCounts.maxMBDublets);

    // Calculate the parallelisation for the triplet finding for this middle
    // spacepoint.
    const int blockSize = std::sqrt(maxBlockSize);
    const dim3 blockSizeFT(blockSize, blockSize);
    const dim3 numBlocksFT(((middleBottomDublets + blockSizeFT.x - 1) /
                            blockSizeFT.x),
                           ((middleTopDublets + blockSizeFT.y - 1) /
                            blockSizeFT.y));

    // Launch the triplet finding for this middle spacepoint.
    kernels::findTriplets<<<numBlocksFT, blockSizeFT>>>(
        // Parameters needed to use all the arrays.
        middleIndex, dubletCounts.maxMBDublets, dubletCounts.maxMTDublets,
        maxTriplets,
        // Parameters of all of the spacepoints.
        nBottomSP, bottomSPArray.get(),
        nMiddleSP, middleSPArray.get(),
        nTopSP, topSPArray.get(),
        // Arrays describing the identified dublets.
        middleBottomCountArray.get(), middleBottomArray.get(),
        middleTopCountArray.get(), middleTopArray.get(),
        // The transformed parameters of the bottom and top spacepoints for
        // spacepoints taking part in dublets.
        bottomSPLinTransArray.get(), topSPLinTransArray.get(),
        // Configuration constants.
        maxScatteringAngle2, sigmaScattering, minHelixDiameter2, pT2perRadius,
        impactMax, impactWeightFactor,
        // Variables storing the results of the triplet finding.
        tripletsPerBottomDubletDevice.get(), tripletIndices.get(),
        objectCountsDevice.get() + MaxTripletsPerSpB,
        objectCountsDevice.get() + AllTriplets, allTripletsArray.get());
    ACTS_CUDA_ERROR_CHECK(cudaGetLastError());
    ACTS_CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    copyToHost(objectCountsHost, objectCountsDevice, NObjectCountTypes);
    allTriplets += objectCountsHost.get()[AllTriplets];

    // Retrieve the maximal number of triplets found for any given bottom-middle
    // dublet.
    int maxTripletsPerSpB = 0;
    ACTS_CUDA_ERROR_CHECK(cudaMemcpy(&maxTripletsPerSpB,
                                     objectCountsDevice.get() +
                                     MaxTripletsPerSpB, sizeof(int),
                                     cudaMemcpyDeviceToHost));
    // If no such triplet has been found, stop here for this middle spacepoint.
    if (maxTripletsPerSpB == 0) {
      continue;
    }

    // Calculate the parallelisation for the "2SpFixed" filtering of the
    // triplets.
    const dim3 blockSizeF2SP(blockSize, blockSize);
    const dim3 numBlocksF2SP(((middleBottomDublets + blockSizeF2SP.x - 1) /
                              blockSizeF2SP.x),
                             ((maxTripletsPerSpB + blockSizeF2SP.y - 1) /
                              blockSizeF2SP.y));

    // Launch the "2SpFixed" filtering of the triplets.
    kernels::filterTriplets2Sp<<<numBlocksF2SP, blockSizeF2SP>>>(
        // Parameters needed to use all the arrays.
        middleIndex, dubletCounts.maxMBDublets, dubletCounts.maxMTDublets,
        middleBottomDublets,
        // Parameters of all of the spacepoints.
        nBottomSP, bottomSPArray.get(),
        nMiddleSP, middleSPArray.get(),
        nTopSP, topSPArray.get(),
        // Variables holding the results of the triplet finding.
        tripletsPerBottomDubletDevice.get(), tripletIndices.get(),
        objectCountsDevice.get() + AllTriplets, allTripletsArray.get(),
        // Configuration constants.
        deltaInvHelixDiameter, deltaRMin, compatSeedWeight, compatSeedLimit,
        // Variables storing the results of the filtering.
        objectCountsDevice.get() + FilteredTriplets,
        filteredTripletsArray.get());
    ACTS_CUDA_ERROR_CHECK(cudaGetLastError());
    ACTS_CUDA_ERROR_CHECK(cudaDeviceSynchronize());


    copyToHost(objectCountsHost, objectCountsDevice, NObjectCountTypes);
    filteredTriplets += objectCountsHost.get()[FilteredTriplets];
  }
  std::cout << "allTriplets = " << allTriplets << ", filteredTriplets = " << filteredTriplets << std::endl;

  /*
  // Calculate the parallelisation for the triplet finding.
  const int numBlocksFT =
      (dubletCounts.nTriplets + maxBlockSize - 1) / maxBlockSize;

  // Create the variables used for the triplet finding. Note that we don't use
  // @c thrust::device_vector for the Triplet array, as that takes much too long
  // with initialising the memory for the vector.
  const std::size_t tripletPerDubletCountsSize = nMiddleSP * nBottomSP;
  auto tripletPerDubletCountsHost =
      make_host_array<int>(tripletPerDubletCountsSize);
  memset(tripletPerDubletCountsHost.get(), 0,
         tripletPerDubletCountsSize * sizeof(int));
  auto tripletPerDubletCountsDevice =
      make_device_array<int>(tripletPerDubletCountsSize);
  copyToDevice(tripletPerDubletCountsDevice, tripletPerDubletCountsHost,
               tripletPerDubletCountsSize);

  auto tripletIndices =
      make_device_array<std::size_t>(nMiddleSP * nBottomSP *
                                     MAX_TRIPLET_PER_MIDDLE_BOTTOM);

  auto tripletCountsHost = make_host_array<int>(nMiddleSP);
  memset(tripletCountsHost.get(), 0, nMiddleSP * sizeof(int));
  auto tripletCountsDevice = make_device_array<int>(nMiddleSP);
  copyToDevice(tripletCountsDevice, tripletCountsHost, nMiddleSP);

  const std::size_t maxTriplets = dubletCounts.nTriplets / (2 * nMiddleSP);
  auto tripletArray =
      make_device_array<Triplet>(nMiddleSP * maxTriplets);

  // Launch the triplet finding.
  kernels::findTriplets<<<numBlocksFT, maxBlockSize>>>(
      dubletCounts.nTriplets, dubletCounts.maxMBDublets,
      dubletCounts.maxMTDublets, maxTriplets,
      nBottomSP, bottomSPArray.get(), nMiddleSP,
      middleSPArray.get(), nTopSP, topSPArray.get(),
      bottomSPLinTransArray.get(), topSPLinTransArray.get(),
      middleBottomCountArray.get(), middleBottomArray.get(),
      middleTopCountArray.get(), middleTopArray.get(), maxScatteringAngle2,
      sigmaScattering, minHelixDiameter2, pT2perRadius, impactMax,
      impactWeightFactor,
      tripletPerDubletCountsDevice.get(), tripletIndices.get(),
      tripletCountsDevice.get(), tripletArray.get());
  ACTS_CUDA_ERROR_CHECK(cudaGetLastError());
  ACTS_CUDA_ERROR_CHECK(cudaDeviceSynchronize());
  */

  // Sort the found triplets, for the filtering to be faster.
  /*
  thrust::device_ptr<Triplet> tripletThrustPtr =
      thrust::device_pointer_cast(tripletArray.get());
  thrust::sort(tripletThrustPtr, tripletThrustPtr + tripletArraySize,
               kernels::TripletSorter());
               */

  // Create the variables used as the output of this filtering.
  /*
  auto tripletCount2SpFiltHostArray = make_host_array<int>(nMiddleSP);
  for (std::size_t i = 0; i < nMiddleSP; ++i) {
    tripletCount2SpFiltHostArray.get()[i] = 0;
  }
  auto tripletCount2SpFiltDeviceArray = make_device_array<int>(nMiddleSP);
  copyToDevice(tripletCount2SpFiltDeviceArray, tripletCount2SpFiltHostArray,
               nMiddleSP);
  auto triplet2SpFiltDeviceArray =
      make_device_array<Triplet>(nMiddleSP * dubletCounts.maxTriplets);
  */

  // Launch the triplet filtering. Note that we only use a single thread per
  // block. This is because the threads will be doing very different operations.
  /*
  kernels::filterTriplets<<<nMiddleSP, 1>>>(
      dubletCounts.maxTriplets, nBottomSP, bottomSPArray.get(), nMiddleSP,
      middleSPArray.get(), nTopSP, topSPArray.get(),
      tripletCountDeviceArray.get(), tripletDeviceArray.get(),
      deltaInvHelixDiameter, deltaRMin, compatSeedWeight, compatSeedLimit);
      */

  return;
}

}  // namespace details
}  // namespace Cuda
}  // namespace Acts
