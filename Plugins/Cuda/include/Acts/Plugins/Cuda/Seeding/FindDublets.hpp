// This file is part of the Acts project.
//
// Copyright (C) 2020 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

// CUDA plugin include(s).
#include "Acts/Plugins/Cuda/Utilities/Arrays.hpp"
#include "Acts/Plugins/Cuda/Utilities/ResultScalar.hpp"

// System include(s).
#include <cstddef>

namespace Acts {
namespace Cuda {
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
                 device_array<int>& middleTopPairs);

}  // namespace details
}  // namespace Cuda
}  // namespace Acts
