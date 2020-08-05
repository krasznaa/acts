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
    transformCoordinates(middleSPArray[middleIndex], bottomSPArray[bottomIndex],
                         ACTS_CUDA_MATRIX2D_ELEMENT(bottomSPLinTransArray,
                                                    nMiddleSP, maxMBDublets,
                                                    middleIndex,
                                                    bottomMatrixIndex),
                         true);
  } else {
    std::size_t topIndex =
        ACTS_CUDA_MATRIX2D_ELEMENT(middleTopArray, nMiddleSP, nTopSP,
                                   middleIndex, topMatrixIndex);
    assert(topIndex < nTopSP);
    transformCoordinates(middleSPArray[middleIndex], topSPArray[topIndex],
                         ACTS_CUDA_MATRIX2D_ELEMENT(topSPLinTransArray,
                                                    nMiddleSP, maxMTDublets,
                                                    middleIndex,
                                                    topMatrixIndex),
                         false);
  }

  return;
}

__global__ void findTriplets(int nTripletCandidates, int maxMBDublets,
                             int maxMTDublets, int maxTriplets,
                             std::size_t nBottomSP,
                             const details::SpacePoint* bottomSPArray,
                             std::size_t nMiddleSP,
                             const details::SpacePoint* middleSPArray,
                             std::size_t nTopSP,
                             const details::SpacePoint* topSPArray,
                             const details::LinCircle* bottomSPLinTransArray,
                             const details::LinCircle* topSPLinTransArray,
                             const int* middleBottomCountArray,
                             const int* middleBottomArray,
                             const int* middleTopCountArray,
                             const int* middleTopArray,
                             float maxScatteringAngle2, float sigmaScattering,
                             float minHelixDiameter2, float pT2perRadius,
                             float impactMax,
                             int* tripletCountArray,
                             details::Triplet* tripletArray) {

  // Get the global index.
  const int tripletIndex = blockIdx.x * blockDim.x + threadIdx.x;

  // If we're out of bounds, finish right away.
  if (tripletIndex >= nTripletCandidates) {
    return;
  }

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
  const std::size_t bottomIndex =
      ACTS_CUDA_MATRIX2D_ELEMENT(middleBottomArray, nMiddleSP, nBottomSP,
                                 middleIndex, bottomMatrixIndex);
  assert(bottomIndex < nBottomSP);
  const std::size_t topIndex =
      ACTS_CUDA_MATRIX2D_ELEMENT(middleTopArray, nMiddleSP, nTopSP,
                                 middleIndex, topMatrixIndex);
  assert(topIndex < nTopSP);

  // Load the transformed coordinates of the bottom spacepoint into the thread.
  const details::LinCircle lb =
      ACTS_CUDA_MATRIX2D_ELEMENT(bottomSPLinTransArray, nMiddleSP, maxMBDublets,
                                 middleIndex, bottomMatrixIndex);

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

  const details::LinCircle lt =
      ACTS_CUDA_MATRIX2D_ELEMENT(topSPLinTransArray, nMiddleSP, maxMTDublets,
                                 middleIndex, topMatrixIndex);

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

  if (Im <= impactMax) {
    const int tripletRow = atomicAdd(tripletCountArray + middleIndex, 1);
    details::Triplet triplet = {bottomIndex, topIndex, Im, B / sqrtf(S2)};
    ACTS_CUDA_MATRIX2D_ELEMENT(tripletArray, nMiddleSP, maxTriplets,
                               middleIndex, tripletRow) = triplet;
  }

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
                  float impactMax) {

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

  // Create the variables used for the triplet finding.
  auto tripletCountDeviceArray = make_device_array<int>(nMiddleSP);
  auto tripletDeviceArray =
      make_device_array<Triplet>(nMiddleSP * dubletCounts.maxTriplets);

  // Launch the triplet finding.
  kernels::findTriplets<<<numBlocksFT, maxBlockSize>>>(
      dubletCounts.nTriplets, dubletCounts.maxMBDublets,
      dubletCounts.maxMTDublets, dubletCounts.maxTriplets,
      nBottomSP, bottomSPArray.get(), nMiddleSP,
      middleSPArray.get(), nTopSP, topSPArray.get(),
      bottomSPLinTransArray.get(), topSPLinTransArray.get(),
      middleBottomCountArray.get(), middleBottomArray.get(),
      middleTopCountArray.get(), middleTopArray.get(), maxScatteringAngle2,
      sigmaScattering, minHelixDiameter2, pT2perRadius, impactMax,
      tripletCountDeviceArray.get(), tripletDeviceArray.get());
  ACTS_CUDA_ERROR_CHECK(cudaGetLastError());
  ACTS_CUDA_ERROR_CHECK(cudaDeviceSynchronize());
  return;
}

}  // namespace details
}  // namespace Cuda
}  // namespace Acts
