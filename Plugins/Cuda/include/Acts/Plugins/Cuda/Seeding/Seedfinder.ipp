// This file is part of the Acts project.
//
// Copyright (C) 2020 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

// CUDA plugin include(s).
#include "Acts/Plugins/Cuda/Seeding/FindDublets.hpp"
#include "Acts/Plugins/Cuda/Seeding/FindTriplets.hpp"
#include "Acts/Plugins/Cuda/Seeding/Types.hpp"
#include "Acts/Plugins/Cuda/Utilities/Arrays.hpp"
#include "Acts/Plugins/Cuda/Utilities/HostMatrix.hpp"
#include "Acts/Plugins/Cuda/Utilities/MemoryManager.hpp"
#include "Acts/Plugins/Cuda/Utilities/ResultScalar.hpp"

// Acts include(s).
#include "Acts/Seeding/InternalSeed.hpp"
#include "Acts/Seeding/InternalSpacePoint.hpp"

// System include(s).
#include <tuple>
#include <vector>
#include <iostream>

namespace Acts {
namespace Cuda {

template <typename external_spacepoint_t>
Seedfinder<external_spacepoint_t>::Seedfinder(
    Acts::SeedfinderConfig<external_spacepoint_t> commonConfig,
    SeedFilterConfig filterConfig)
    : m_commonConfig(std::move(commonConfig)), m_filterConfig(filterConfig) {
  // calculation of scattering using the highland formula
  // convert pT to p once theta angle is known
  m_commonConfig.highland = 13.6 * std::sqrt(m_commonConfig.radLengthPerSeed) *
                      (1 + 0.038 * std::log(m_commonConfig.radLengthPerSeed));
  float maxScatteringAngle = m_commonConfig.highland / m_commonConfig.minPt;
  m_commonConfig.maxScatteringAngle2 = maxScatteringAngle * maxScatteringAngle;

  // helix radius in homogeneous magnetic field. Units are Kilotesla, MeV and
  // millimeter
  // TODO: change using ACTS units
  m_commonConfig.pTPerHelixRadius = 300. * m_commonConfig.bFieldInZ;
  m_commonConfig.minHelixDiameter2 =
      std::pow(m_commonConfig.minPt * 2 / m_commonConfig.pTPerHelixRadius, 2);
  m_commonConfig.pT2perRadius =
      std::pow(m_commonConfig.highland / m_commonConfig.pTPerHelixRadius, 2);
}

template <typename external_spacepoint_t>
template <typename sp_range_t>
std::vector<Seed<external_spacepoint_t>>
Seedfinder<external_spacepoint_t>::createSeedsForGroup(
    sp_range_t bottomSPs, sp_range_t middleSPs, sp_range_t topSPs) const {

  // Create the result vector right away.
  std::vector<Seed<external_spacepoint_t>> outputVec;

  //---------------------------------
  // Matrix Flattening
  //---------------------------------

  // Reset all the memory allocated for Acts on the CUDA device.
  MemoryManager::instance().reset();

  // Create more convenient vectors out of the space point containers.
  auto spVecMaker = [](sp_range_t spRange) {
    std::vector<const Acts::InternalSpacePoint<external_spacepoint_t>*> result;
    for (auto* sp : spRange) {
      result.push_back(sp);
    }
    return result;
  };

  std::vector<const Acts::InternalSpacePoint<external_spacepoint_t>*>
      bottomSPVec(spVecMaker(bottomSPs));
  std::vector<const Acts::InternalSpacePoint<external_spacepoint_t>*>
      middleSPVec(spVecMaker(middleSPs));
  std::vector<const Acts::InternalSpacePoint<external_spacepoint_t>*>
      topSPVec(spVecMaker(topSPs));

  // If either one of them is empty, we have nothing to find.
  if ((middleSPVec.size() == 0) || (bottomSPVec.size() == 0) ||
      (topSPVec.size() == 0)) {
    return outputVec;
  }

  // Create helper objects for storing information about the spacepoints on the
  // host in single memory blobs.
  auto bottomSPArray = make_host_array<details::SpacePoint>(bottomSPVec.size());
  auto middleSPArray = make_host_array<details::SpacePoint>(middleSPVec.size());
  auto topSPArray = make_host_array<details::SpacePoint>(topSPVec.size());

  // Fill these memory blobs.
  auto fillSPArray = [](details::SpacePoint* array, const auto& spVec) {
    for (std::size_t i = 0; i < spVec.size(); ++i) {
      array[i].x = spVec[i]->x();
      array[i].y = spVec[i]->y();
      array[i].z = spVec[i]->z();
      array[i].radius = spVec[i]->radius();
      array[i].varianceR = spVec[i]->varianceR();
      array[i].varianceZ = spVec[i]->varianceZ();
    }
  };
  fillSPArray(bottomSPArray.get(), bottomSPVec);
  fillSPArray(middleSPArray.get(), middleSPVec);
  fillSPArray(topSPArray.get(), topSPVec);

  // Copy the memory blobs to the device.
  auto bottomSPDeviceArray =
      make_device_array<details::SpacePoint>(bottomSPVec.size());
  auto middleSPDeviceArray =
      make_device_array<details::SpacePoint>(middleSPVec.size());
  auto topSPDeviceArray =
      make_device_array<details::SpacePoint>(topSPVec.size());
  copyToDevice(bottomSPDeviceArray, bottomSPArray, bottomSPVec.size());
  copyToDevice(middleSPDeviceArray, middleSPArray, middleSPVec.size());
  copyToDevice(topSPDeviceArray, topSPArray, topSPVec.size());

  //---------------------------------
  // GPU Execution
  //---------------------------------

  // Matrices holding the viable bottom-middle and middle-top pairs.
  HostMatrix<1, int> middleBottomCounts({middleSPVec.size()});
  HostMatrix<1, int> middleTopCounts({middleSPVec.size()});

  // Reset the values in the count vectors.
  for (std::size_t i = 0; i < middleSPVec.size(); ++i) {
    middleBottomCounts.set({i}, 0);
    middleTopCounts.set({i}, 0);
  }

  // Set up the device memory for these.
  auto middleBottomCountArray =
      make_device_array<int>(middleBottomCounts.totalSize());
  auto middleBottomArray =
      make_device_array<int>(middleSPVec.size() * bottomSPVec.size());
  auto middleTopCountArray =
      make_device_array<int>(middleTopCounts.totalSize());
  auto middleTopArray =
      make_device_array<int>(middleSPVec.size() * topSPVec.size());
  middleBottomCounts.copyTo(middleBottomCountArray);
  middleTopCounts.copyTo(middleTopCountArray);

  // Launch the dublet finding code.
  details::findDublets(m_commonConfig.maxBlockSize,
                       bottomSPVec.size(), bottomSPDeviceArray,
                       middleSPVec.size(), middleSPDeviceArray,
                       topSPVec.size(), topSPDeviceArray,
                       m_commonConfig.deltaRMin, m_commonConfig.deltaRMax,
                       m_commonConfig.cotThetaMax, m_commonConfig.collisionRegionMin,
                       m_commonConfig.collisionRegionMax,
                       middleBottomCountArray, middleBottomArray,
                       middleTopCountArray, middleTopArray);

  // Count the number of dublets that we have to launch the subsequent steps
  // for.
  details::DubletCounts dubletCounts =
      details::countDublets(m_commonConfig.maxBlockSize, middleSPVec.size(),
                            middleBottomCountArray, middleTopCountArray);

  // If no dublets/triplet candidates have been found, stop here.
  if ((dubletCounts.nDublets == 0) || (dubletCounts.nTriplets == 0)) {
    return outputVec;
  }

  // Launch the triplet finding code on all of the previously found dublets.
  details::findTriplets(m_commonConfig.maxBlockSize, dubletCounts,
                        bottomSPVec.size(), bottomSPDeviceArray,
                        middleSPVec.size(), middleSPDeviceArray,
                        topSPVec.size(), topSPDeviceArray,
                        middleBottomCountArray, middleBottomArray,
                        middleTopCountArray, middleTopArray,
                        m_commonConfig.maxScatteringAngle2, m_commonConfig.sigmaScattering,
                        m_commonConfig.minHelixDiameter2, m_commonConfig.pT2perRadius,
                        m_commonConfig.impactMax, m_filterConfig.impactWeightFactor,
                        m_filterConfig.deltaInvHelixDiameter, m_filterConfig.deltaRMin,
                        m_filterConfig.compatSeedWeight, m_filterConfig.compatSeedLimit);

  return outputVec;
}

}  // namespace Cuda
}  // namespace Acts
