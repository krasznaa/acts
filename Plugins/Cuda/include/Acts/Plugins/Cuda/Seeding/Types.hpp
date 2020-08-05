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
  int bIndex;
  int tIndex;
  float topRadius;
  float impactParameter;
  float invHelixDiameter;
  float weight;
};  // struct Triplet

/// Helper struct summarising the results of the dublet search
struct DubletCounts {
  int nDublets = 0; ///< The total number of dublets (M-B and M-T) found
  int nTriplets = 0; ///< The total number of triplet candidates found
  int maxMBDublets = 0; ///< The maximal number of middle-bottom dublets
  int maxMTDublets = 0; ///< The maximal number of middle-top dublets
}; // struct DubletCounts

/// Number of variables stored in the internal matrices per spacepoint
static constexpr std::size_t SP_DIMENSIONS = 6;
/// Index of the spacepoint X coordinates in the internal matrices
static constexpr std::size_t SP_X_INDEX = 0;
/// Index of the spacepoint Y coordinates in the internal matrices
static constexpr std::size_t SP_Y_INDEX = 1;
/// Index of the spacepoint Z coordinates in the internal matrices
static constexpr std::size_t SP_Z_INDEX = 2;
/// Index of the spacepoint radious coordinates in the internal matrices
static constexpr std::size_t SP_R_INDEX = 3;
/// Index of the spacepoint R-variance coordinates in the internal matrices
static constexpr std::size_t SP_VR_INDEX = 4;
/// Index of the spacepoint Z-variance coordinates in the internal matrices
static constexpr std::size_t SP_VZ_INDEX = 5;

/// Number of linearly transformed variables stored in the internal matrices per
/// spacepoint
static constexpr std::size_t LC_DIMENSIONS = 6;
static constexpr std::size_t LC_ZO_INDEX = 0;
static constexpr std::size_t LC_CT_INDEX = 1;
static constexpr std::size_t LC_DR_INDEX = 2;
static constexpr std::size_t LC_ER_INDEX = 3;
static constexpr std::size_t LC_U_INDEX = 4;
static constexpr std::size_t LC_V_INDEX = 5;

}  // namespace details
}  // namespace Cuda
}  // namespace Acts
