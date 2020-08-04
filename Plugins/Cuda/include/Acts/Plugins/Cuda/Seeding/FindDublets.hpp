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
                 device_array<int>& middleBottomCountArray,
                 device_array<int>& middleBottomArray,
                 device_array<int>& middleTopCountArray,
                 device_array<int>& middleTopArray);

}  // namespace details
}  // namespace Cuda
}  // namespace Acts
