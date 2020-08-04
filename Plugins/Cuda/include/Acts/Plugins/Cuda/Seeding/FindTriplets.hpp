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

int countTriplets(int maxBlockSize, std::size_t nMiddleSP,
                  const device_array<int>& middleBottomCountArray,
                  const device_array<int>& middleTopCountArray);

void findTriplets(int maxBlockSize, int nTripletCandidates,
                  std::size_t nBottomSP,
                  const device_array<float>& bottomSPDeviceMatrix,
                  std::size_t nMiddleSP,
                  const device_array<float>& middleSPDeviceMatrix,
                  std::size_t nTopSP,
                  const device_array<float>& topSPDeviceMatrix,
                  const device_array<int>& middleBottomCountArray,
                  const device_array<int>& middleBottomArray,
                  const device_array<int>& middleTopCountArray,
                  const device_array<int>& middleTopArray);

}  // namespace details
}  // namespace Cuda
}  // namespace Acts
