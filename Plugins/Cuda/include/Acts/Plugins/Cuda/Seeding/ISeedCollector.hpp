// This file is part of the Acts project.
//
// Copyright (C) 2020 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

// CUDA plugin include(s).
#include "Acts/Plugins/Cuda/Seeding/Types.hpp"
#include "Acts/Plugins/Cuda/Utilities/Arrays.hpp"

namespace Acts {
namespace Cuda {
namespace details {

/// Interface for the class used for collecting seeds from the GPU code
class ISeedCollector {

 public:
  /// Virtual destructor, to make vtable happy
  virtual ~ISeedCollector() = default;

  /// Collect the triplets that passed the "2SpFixed" selection already
  virtual void collectTriplets(std::size_t nTriplets,
                               const host_array<Triplet>& triplets,
                               std::size_t middleIndex, float zOrigin) = 0;

};  // class ISeedCollector

}  // namespace details
}  // namespace Cuda
}  // namespace Acts
