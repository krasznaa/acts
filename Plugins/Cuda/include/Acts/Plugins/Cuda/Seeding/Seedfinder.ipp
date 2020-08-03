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
#include "Acts/Plugins/Cuda/Seeding/Types.hpp"
#include "Acts/Plugins/Cuda/Utilities/Arrays.hpp"
#include "Acts/Plugins/Cuda/Utilities/HostMatrix.hpp"
#include "Acts/Plugins/Cuda/Utilities/ResultScalar.hpp"

// Acts include(s).
#include "Acts/Seeding/InternalSeed.hpp"
#include "Acts/Seeding/InternalSpacePoint.hpp"

namespace Acts {
namespace Cuda {

template <typename external_spacepoint_t>
Seedfinder<external_spacepoint_t>::Seedfinder(
    Acts::SeedfinderConfig<external_spacepoint_t> config)
    : m_config(std::move(config)) {
  // calculation of scattering using the highland formula
  // convert pT to p once theta angle is known
  m_config.highland = 13.6 * std::sqrt(m_config.radLengthPerSeed) *
                      (1 + 0.038 * std::log(m_config.radLengthPerSeed));
  float maxScatteringAngle = m_config.highland / m_config.minPt;
  m_config.maxScatteringAngle2 = maxScatteringAngle * maxScatteringAngle;

  // helix radius in homogeneous magnetic field. Units are Kilotesla, MeV and
  // millimeter
  // TODO: change using ACTS units
  m_config.pTPerHelixRadius = 300. * m_config.bFieldInZ;
  m_config.minHelixDiameter2 =
      std::pow(m_config.minPt * 2 / m_config.pTPerHelixRadius, 2);
  m_config.pT2perRadius =
      std::pow(m_config.highland / m_config.pTPerHelixRadius, 2);
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

  // Create more convenient vectors out of the space point containers.
  auto spVecMaker = [](sp_range_t spRange) {
    std::vector<const Acts::InternalSpacePoint<external_spacepoint_t>*> result;
    for (auto* sp : spRange) {
      result.push_back(sp);
    }
    return result;
  };

  std::vector<const Acts::InternalSpacePoint<external_spacepoint_t>*>
      bottomSPvec(spVecMaker(bottomSPs));
  std::vector<const Acts::InternalSpacePoint<external_spacepoint_t>*>
      middleSPvec(spVecMaker(middleSPs));
  std::vector<const Acts::InternalSpacePoint<external_spacepoint_t>*>
      topSPvec(spVecMaker(topSPs));

  // If either one of them is empty, we have nothing to find.
  if ((middleSPvec.size() == 0) || (bottomSPvec.size() == 0) ||
      (topSPvec.size() == 0)) {
    return outputVec;
  }

  // Create helper objects for creating the memory describing the space points,
  // here on the host.
  HostMatrix<2, float> bottomSPHostMatrix({bottomSPvec.size(),
                                           details::SP_DIMENSIONS});
  HostMatrix<2, float> middleSPHostMatrix({middleSPvec.size(),
                                           details::SP_DIMENSIONS});
  HostMatrix<2, float> topSPHostMatrix({topSPvec.size(),
                                        details::SP_DIMENSIONS});

  // Fill them with information coming from the Acts space point objects.
  auto fillSPMatrix = [](auto& matrix, const auto& spVec) {
    for (std::size_t i = 0; i < spVec.size(); ++i) {
      matrix.set({i, details::SP_X_INDEX}, spVec[i]->x());
      matrix.set({i, details::SP_Y_INDEX}, spVec[i]->y());
      matrix.set({i, details::SP_Z_INDEX}, spVec[i]->z());
      matrix.set({i, details::SP_R_INDEX}, spVec[i]->radius());
      matrix.set({i, details::SP_VR_INDEX}, spVec[i]->varianceR());
      matrix.set({i, details::SP_VZ_INDEX}, spVec[i]->varianceZ());
    }
  };
  fillSPMatrix(bottomSPHostMatrix, bottomSPvec);
  fillSPMatrix(middleSPHostMatrix, middleSPvec);
  fillSPMatrix(topSPHostMatrix, topSPvec);

  // Copy the memory blocks to the device.
  auto bottomSPDeviceMatrix =
    make_device_array<float>(bottomSPHostMatrix.totalSize());
  auto middleSPDeviceMatrix =
    make_device_array<float>(middleSPHostMatrix.totalSize());
  auto topSPDeviceMatrix =
    make_device_array<float>(topSPHostMatrix.totalSize());
  bottomSPHostMatrix.copyTo(bottomSPDeviceMatrix);
  middleSPHostMatrix.copyTo(middleSPDeviceMatrix);
  topSPHostMatrix.copyTo(topSPDeviceMatrix);

  //---------------------------------
  // GPU Execution
  //---------------------------------

  // Matrices holding the viable bottom-middle and middle-top pairs on the
  // device.
  auto bottomMiddlePairs = make_device_array<int>(2 * bottomSPvec.size() *
                                                  middleSPvec.size());
  auto middleTopPairs = make_device_array<int>(2 * middleSPvec.size() *
                                               topSPvec.size());
  // The number of elements filled into these matrices.
  ResultScalar<int> nBottomMiddlePairs;
  ResultScalar<int> nMiddleTopPairs;

  // Launch the "triplet flagging" code.
  details::findDublets(m_config.maxBlockSize,
                       bottomSPvec.size(), bottomSPDeviceMatrix,
                       middleSPvec.size(), middleSPDeviceMatrix,
                       topSPvec.size(), topSPDeviceMatrix,
                       m_config.deltaRMin, m_config.deltaRMax,
                       m_config.cotThetaMax, m_config.collisionRegionMin,
                       m_config.collisionRegionMax,
                       nBottomMiddlePairs, bottomMiddlePairs,
                       nMiddleTopPairs, middleTopPairs);

  return outputVec;
}

}  // namespace Cuda
}  // namespace Acts
