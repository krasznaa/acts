// This file is part of the Acts project.
//
// Copyright (C) 2020 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

// System include(s).
#include <cassert>

namespace Acts {
namespace Cuda {
namespace details {

template <typename external_spacepoint_t>
SeedCollector<external_spacepoint_t>::SeedCollector(
  const SeedfinderConfig<external_spacepoint_t>& config,
  const SPVec_t& bottomSPVec, const SPVec_t& middleSPVec,
  const SPVec_t& topSPVec)
  : m_config(&config), m_bottomSPVec(&bottomSPVec), m_middleSPVec(&middleSPVec),
    m_topSPVec(&topSPVec) {

}

template <typename external_spacepoint_t>
void SeedCollector<external_spacepoint_t>::collectTriplets(
   std::size_t nTriplets, const host_array<Triplet>& triplets,
   std::size_t middleIndex, float zOrigin) {

  // A sanity check.
  assert(middleIndex < m_middleSPVec->size());

  // Create InternalSeed objects out of the triplets.
  std::vector<std::pair<
    float, std::unique_ptr<const InternalSeed<external_spacepoint_t>>>>
      seedsPerSpM;
  for (std::size_t i = 0; i < nTriplets; ++i) {
    assert(triplets.get()[i].bottomIndex < m_bottomSPVec->size());
    assert(triplets.get()[i].topIndex < m_topSPVec->size());
    seedsPerSpM.push_back(std::make_pair(
      triplets.get()[i].weight,
      std::make_unique<const InternalSeed<external_spacepoint_t>>(
        *((*m_bottomSPVec)[triplets.get()[i].bottomIndex]),
        *((*m_middleSPVec)[middleIndex]),
        *((*m_topSPVec)[triplets.get()[i].topIndex]), zOrigin)));
  }

  // Use the CPU filter to select just the final seeds.
  m_config->seedFilter->filterSeeds_1SpFixed(seedsPerSpM, m_seeds);
  return;
}

template <typename external_spacepoint_t>
std::vector<Seed<external_spacepoint_t> >
SeedCollector<external_spacepoint_t>::getSeeds() const {

   return m_seeds;
}

}  // namespace details
}  // namespace Cuda
}  // namespace Acts
