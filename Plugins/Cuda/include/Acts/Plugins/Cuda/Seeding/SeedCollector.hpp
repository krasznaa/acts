// This file is part of the Acts project.
//
// Copyright (C) 2020 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

// CUDA plugin include(s).
#include "Acts/Plugins/Cuda/Seeding/ISeedCollector.hpp"
#include "Acts/Plugins/Cuda/Seeding/Types.hpp"
#include "Acts/Plugins/Cuda/Utilities/Arrays.hpp"

// Acts include(s).
#include "Acts/Seeding/InternalSpacePoint.hpp"
#include "Acts/Seeding/Seed.hpp"
#include "Acts/Seeding/SeedfinderConfig.hpp"

// System include(s).
#include <vector>

namespace Acts {
namespace Cuda {
namespace details {

/// Type used by the CUDA code to pass triplet candidates back to the host
template <typename external_spacepoint_t>
class SeedCollector : public virtual ISeedCollector {

 public:
  /// Type for the internal spacepoint vectors
  using SPVec_t =
    std::vector<const Acts::InternalSpacePoint<external_spacepoint_t>*>;

  /// Constructor with all necessary parameters
  SeedCollector(const SeedfinderConfig<external_spacepoint_t>& config,
                const SPVec_t& bottomSPVec, const SPVec_t& middleSPVec,
                const SPVec_t& topSPVec);

  /// Collect all triplets for a single middle spacepoint
  virtual void collectTriplets(std::size_t nTriplets,
                               const host_array<Triplet>& triplets,
                               std::size_t middleIndex, float zOrigin) override;

  /// Get the seeds collected from the GPU code
  std::vector<Seed<external_spacepoint_t> > getSeeds() const;

 private:
  /// Pointer to the parent's configuration
  const SeedfinderConfig<external_spacepoint_t>* m_config;
  /// Pointer to the parent's bottom spacepoint vector
  const SPVec_t* m_bottomSPVec;
  /// Pointer to the parent's middle spacepoint vector
  const SPVec_t* m_middleSPVec;
  /// Pointer to the parent's top spacepoint vector
  const SPVec_t* m_topSPVec;
  /// List of seeds collected
  std::vector<Seed<external_spacepoint_t> > m_seeds;

};  // class SeedCollector

}  // namespace details
}  // namespace Cuda
}  // namespace Acts

// Include the implementation.
#include "SeedCollector.ipp"
