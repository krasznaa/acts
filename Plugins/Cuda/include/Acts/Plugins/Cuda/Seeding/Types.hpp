// This file is part of the Acts project.
//
// Copyright (C) 2020 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

// System include(s).
#include <cstddef>

namespace Acts {
namespace Cuda {
namespace details {

/// Structure used in the CUDA-based triplet finding
struct Triplet {
  std::size_t bottomIndex = 0;
  std::size_t topIndex = 0;
  float impactParameter = 0.0f;
  float curvature = 0.0f;
};  // struct Triplet

/// Helper struct describing a spacepoint on the device
struct SpacePoint {
  float x = 0.0f; ///< x-coordinate in beam system coordinates
  float y = 0.0f; ///< y-coordinate in beam system coordinates
  float z = 0.0f; ///< z-coordinate in beam system coordinates
  float radius = 0.0f; ///< radius in beam system coordinates
  float varianceR = 0.0f;
  float varianceZ = 0.0f;
};

/// Helper struct summarising the results of the dublet search
struct DubletCounts {
  int nDublets = 0; ///< The total number of dublets (M-B and M-T) found
  int nTriplets = 0; ///< The total number of triplet candidates found
  int maxMBDublets = 0; ///< The maximal number of middle-bottom dublets
  int maxMTDublets = 0; ///< The maximal number of middle-top dublets
  int maxTriplets = 0; ///< The maximal number of triplets for any middle SP
}; // struct DubletCounts

/// Helper struct holding the linearly transformed coordinates of spacepoints
struct LinCircle {
  float Zo = 0.0f;
  float cotTheta = 0.0f;
  float iDeltaR = 0.0f;
  float Er = 0.0f;
  float U = 0.0f;
  float V = 0.0f;
};  // struct LinCircle

}  // namespace details
}  // namespace Cuda
}  // namespace Acts
