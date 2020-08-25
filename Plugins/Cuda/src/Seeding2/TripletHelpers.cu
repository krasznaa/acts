// This file is part of the Acts project.
//
// Copyright (C) 2020 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

// CUDA plugin include(s).
#include "TripletHelpers.cuh"

// System include(s).
#include <cassert>

namespace Acts {
namespace Cuda {
namespace Details {

__host__ __device__ std::size_t countTripletsToEvaluate(
    std::size_t middleIndexStart, std::size_t nParallelMiddleSPs,
    const unsigned int* middleBottomDubletCounts,
    const unsigned int* middleTopDubletCounts) {
  // A small security check.
  assert(nParallelMiddleSPs > 0);

  // Do the calculation.
  std::size_t result = 0;
  const std::size_t middleIndexEnd = middleIndexStart + nParallelMiddleSPs;
  for (std::size_t i = middleIndexStart; i < middleIndexEnd; ++i) {
    result += middleBottomDubletCounts[i] * middleTopDubletCounts[i];
  }
  return result;
}

__host__ __device__ TripletFinderIndices findTripletIndex(
    std::size_t i, std::size_t middleIndexStart, std::size_t nParallelMiddleSPs,
    const unsigned int* middleBottomDubletCounts,
    const unsigned int* middleTopDubletCounts) {
  // Create the result obejct.
  TripletFinderIndices result;

  // Find which middle spacepoint the index refers to.
  int helperIndex1 = i, helperIndex2 = 0;
  unsigned int helperCount =
      middleBottomDubletCounts[middleIndexStart + helperIndex2] *
      middleTopDubletCounts[middleIndexStart + helperIndex2];
  while (helperIndex1 >= helperCount) {
    helperIndex1 -= helperCount;
    ++helperIndex2;
    helperCount = middleBottomDubletCounts[middleIndexStart + helperIndex2] *
                  middleTopDubletCounts[middleIndexStart + helperIndex2];
  }
  assert(helperIndex1 >= 0);
  assert(helperIndex2 < nParallelMiddleSPs);
  result.middleIndex = middleIndexStart + helperIndex2;

  // Find which middle-bottom and middle-top indices the index refers to.
  result.bottomDubletIndex =
      helperIndex1 / middleTopDubletCounts[result.middleIndex];
  assert(result.bottomDubletIndex <
         middleBottomDubletCounts[result.middleIndex]);
  result.topDubletIndex =
      (helperIndex1 -
       result.bottomDubletIndex * middleTopDubletCounts[result.middleIndex]);
  assert(result.topDubletIndex < middleTopDubletCounts[result.middleIndex]);

  // Return the indices.
  return result;
}

}  // namespace Details
}  // namespace Cuda
}  // namespace Acts
