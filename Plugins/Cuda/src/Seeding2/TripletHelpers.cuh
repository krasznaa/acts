// This file is part of the Acts project.
//
// Copyright (C) 2020 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

namespace Acts {
namespace Cuda {
namespace Details {

/// Helper function counting how many triplet combinations to evaluate with the
/// triplet finding kernel.
__host__ __device__ std::size_t countTripletsToEvaluate(
    std::size_t middleIndexStart, std::size_t nParallelMiddleSPs,
    const unsigned int* middleBottomDubletCounts,
    const unsigned int* middleTopDubletCounts);

/// Struct returning the indices of the objects to use in the triplet finding
struct TripletFinderIndices {
  /// Index of the middle spacepoint
  std::size_t middleIndex = 0;
  /// Index of the middle-bottom dublet
  std::size_t bottomDubletIndex = 0;
  /// Index of the middle-top dublet
  std::size_t topDubletIndex = 0;
};

/// Helper function figuring out which triplet combination to evaluate in a
/// given device thread.
__host__ __device__ TripletFinderIndices findTripletIndex(
    std::size_t i, std::size_t middleIndexStart, std::size_t nParallelMiddleSPs,
    const unsigned int* middleBottomDubletCounts,
    const unsigned int* middleTopDubletCounts);

}  // namespace Details
}  // namespace Cuda
}  // namespace Acts
