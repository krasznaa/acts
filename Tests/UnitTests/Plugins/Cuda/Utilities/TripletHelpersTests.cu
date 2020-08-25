// This file is part of the Acts project.
//
// Copyright (C) 2020 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

// CUDA plugin include(s).
#include "../../../../../Plugins/Cuda/src/Seeding2/TripletHelpers.cuh"
#include "../../../../../Plugins/Cuda/src/Utilities/MatrixMacros.hpp"

// Boost include(s).
#include <boost/test/unit_test.hpp>

// System include(s).
#include <vector>

namespace Acts {
namespace Test {
namespace Cuda {

// Create test "count vectors".
static const std::vector<unsigned int> middleBottomDubletCounts{2, 0, 5, 2,
                                                                7, 3, 3, 6};
static const std::vector<unsigned int> middleTopDubletCounts{5, 5, 11, 3,
                                                             7, 2, 9,  13};
static const std::vector<std::vector<unsigned int>> tripletsPerBottomDublet{
    {1, 0},    {},        {2, 0, 0, 1, 1},   {0, 1}, {0, 2, 2, 1, 1, 0, 0},
    {1, 0, 1}, {0, 0, 1}, {4, 4, 0, 0, 1, 0}};

// Create helper variables for various counts.
static constexpr std::size_t nMiddleSPs = 8;
static constexpr std::size_t maxBottomDublets = 7;
static constexpr std::size_t maxTopDublets = 13;

BOOST_AUTO_TEST_SUITE(TripletHelpers)

BOOST_AUTO_TEST_CASE(countTripletsToEvaluate) {
  // Perform some simple tests with the hardcoded vectors.
  BOOST_REQUIRE(Acts::Cuda::Details::countTripletsToEvaluate(
                    0, 2, middleBottomDubletCounts.data(),
                    middleTopDubletCounts.data()) == 10);
  BOOST_REQUIRE(Acts::Cuda::Details::countTripletsToEvaluate(
                    0, 3, middleBottomDubletCounts.data(),
                    middleTopDubletCounts.data()) == 65);
  BOOST_REQUIRE(Acts::Cuda::Details::countTripletsToEvaluate(
                    3, 2, middleBottomDubletCounts.data(),
                    middleTopDubletCounts.data()) == 55);
}

BOOST_AUTO_TEST_CASE(tripletFinderIndex) {
  // Perform some simple tests with the hardcoded vectors.
  auto index = Acts::Cuda::Details::tripletFinderIndex(
      0, 0, 2, middleBottomDubletCounts.data(), middleTopDubletCounts.data());
  BOOST_REQUIRE(index.middleIndex == 0);
  BOOST_REQUIRE(index.bottomDubletIndex == 0);
  BOOST_REQUIRE(index.topDubletIndex == 0);

  index = Acts::Cuda::Details::tripletFinderIndex(
      15, 0, 3, middleBottomDubletCounts.data(), middleTopDubletCounts.data());
  BOOST_REQUIRE(index.middleIndex == 2);
  BOOST_REQUIRE(index.bottomDubletIndex == 0);
  BOOST_REQUIRE(index.topDubletIndex == 5);
}

BOOST_AUTO_TEST_CASE(combinedTripletFinders) {
  // Set up a 3-dimensional matrix (using std::vector) that we use to test that
  // every index would be iterated over using the helper functions.
  std::vector<std::vector<std::vector<int>>> testMatrix(
      nMiddleSPs, std::vector<std::vector<int>>(
                      maxBottomDublets, std::vector<int>(maxTopDublets, 0)));

  // Fill up a reference of what the test matrix should end up looking like in
  // the end.
  auto referenceMatrix = testMatrix;
  for (std::size_t i = 0; i < nMiddleSPs; ++i) {
    for (std::size_t j = 0; j < middleBottomDubletCounts.at(i); ++j) {
      for (std::size_t k = 0; k < middleTopDubletCounts.at(i); ++k) {
        referenceMatrix.at(i).at(j).at(k) = 1;
      }
    }
  }

  // Iterate over all triplet combinations, and check that we would reach every
  // element of the 3D matrix that we had to.
  const std::size_t nTriplets = Acts::Cuda::Details::countTripletsToEvaluate(
      0, nMiddleSPs, middleBottomDubletCounts.data(),
      middleTopDubletCounts.data());
  for (std::size_t i = 0; i < nTriplets; ++i) {
    auto ti = Acts::Cuda::Details::tripletFinderIndex(
        i, 0, nMiddleSPs, middleBottomDubletCounts.data(),
        middleTopDubletCounts.data());
    testMatrix.at(ti.middleIndex)
        .at(ti.bottomDubletIndex)
        .at(ti.topDubletIndex) = 1;
  }

  // Check whether the two match.
  BOOST_REQUIRE(testMatrix == referenceMatrix);
}

BOOST_AUTO_TEST_CASE(tripletFilterIndex) {
  // Set up a 2-dimensional matrix with the contents of
  // "tripletsPerBottomDublet".
  std::vector<unsigned int> tripletsPerBottomDubletArray(
      nMiddleSPs * maxBottomDublets, 0);
  for (std::size_t i = 0; i < nMiddleSPs; ++i) {
    for (std::size_t j = 0; j < middleBottomDubletCounts.at(i); ++j) {
      ACTS_CUDA_MATRIX2D_ELEMENT(tripletsPerBottomDubletArray.data(),
                                 nMiddleSPs, maxBottomDublets, i, j) =
          tripletsPerBottomDublet.at(i).at(j);
    }
  }

  // Perform some simple tests with the hardcoded vectors.
  auto tfi = Acts::Cuda::Details::tripletFilterIndex(
      0, 0, nMiddleSPs, maxBottomDublets, middleBottomDubletCounts.data(),
      tripletsPerBottomDubletArray.data());
  BOOST_REQUIRE(tfi.middleIndex == 0);
  BOOST_REQUIRE(tfi.bottomDubletIndex == 0);
  BOOST_REQUIRE(tfi.tripletIndex == 0);
  tfi = Acts::Cuda::Details::tripletFilterIndex(
      10, 0, nMiddleSPs, maxBottomDublets, middleBottomDubletCounts.data(),
      tripletsPerBottomDubletArray.data());
  BOOST_REQUIRE(tfi.middleIndex == 4);
  BOOST_REQUIRE(tfi.bottomDubletIndex == 3);
  BOOST_REQUIRE(tfi.tripletIndex == 0);

  // Set up a 3-dimensional matrix (using std::vector) that we use to test that
  // every index would be iterated over using the helper function.
  std::vector<std::vector<std::vector<int>>> testMatrix(
      nMiddleSPs, std::vector<std::vector<int>>(
                      maxBottomDublets, std::vector<int>(maxTopDublets, 0)));

  // Fill up a reference of what the test matrix should end up looking like in
  // the end.
  auto referenceMatrix = testMatrix;
  for (std::size_t i = 0; i < nMiddleSPs; ++i) {
    for (std::size_t j = 0; j < middleBottomDubletCounts.at(i); ++j) {
      for (std::size_t k = 0; k < tripletsPerBottomDublet.at(i).at(j); ++k) {
        referenceMatrix.at(i).at(j).at(k) = 1;
      }
    }
  }

  // Now fill the test matrix using the helper function. Note that the number
  // of triplets to evaluate is hardcoded here. In the real code the count is
  // collected during the triplet finding.
  static constexpr std::size_t allTriplets = 24;
  for (std::size_t i = 0; i < allTriplets; ++i) {
    tfi = Acts::Cuda::Details::tripletFilterIndex(
        i, 0, nMiddleSPs, maxBottomDublets, middleBottomDubletCounts.data(),
        tripletsPerBottomDubletArray.data());
    testMatrix.at(tfi.middleIndex)
        .at(tfi.bottomDubletIndex)
        .at(tfi.tripletIndex) = 1;
  }

  // Check whether the two match.
  BOOST_REQUIRE(testMatrix == referenceMatrix);
}

BOOST_AUTO_TEST_SUITE_END()

}  // namespace Cuda
}  // namespace Test
}  // namespace Acts
