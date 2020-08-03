// This file is part of the Acts project.
//
// Copyright (C) 2020 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

// Acts include(s).
#include "Acts/Seeding/Seed.hpp"
#include "Acts/Seeding/SeedfinderConfig.hpp"

namespace Acts {
namespace Cuda {

template <typename external_spacepoint_t>
class Seedfinder {
  ///////////////////////////////////////////////////////////////////
  // Public methods:
  ///////////////////////////////////////////////////////////////////

 public:
  Seedfinder(SeedfinderConfig<external_spacepoint_t> config);

  ~Seedfinder() = default;
  /**    @name Disallow default instantiation, copy, assignment */
  //@{
  Seedfinder() = delete;
  Seedfinder(const Seedfinder<external_spacepoint_t>&) = delete;
  Seedfinder<external_spacepoint_t>& operator=(
      const Seedfinder<external_spacepoint_t>&) = delete;
  //@}

  /// Create all seeds from the space points in the three iterators.
  /// Can be used to parallelize the seed creation
  /// @param bottom group of space points to be used as innermost SP in a seed.
  /// @param middle group of space points to be used as middle SP in a seed.
  /// @param top group of space points to be used as outermost SP in a seed.
  /// Ranges must return pointers.
  /// Ranges must be separate objects for each parallel call.
  /// @return vector in which all found seeds for this group are stored.
  template <typename sp_range_t>
  std::vector<Seed<external_spacepoint_t> > createSeedsForGroup(
      sp_range_t bottomSPs, sp_range_t middleSPs, sp_range_t topSPs) const;

 private:
  /// Configuration for the seed finder
  SeedfinderConfig<external_spacepoint_t> m_config;
};

}  // namespace Cuda
}  // namespace Acts

// Include the template implementation.
#include "Acts/Plugins/Cuda/Seeding/Seedfinder.ipp"
