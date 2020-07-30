// This file is part of the Acts project.
//
// Copyright (C) 2020 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

// CUDA plugin include(s).
#include "Acts/Plugins/Cuda/Utilities/HostMemory.hpp"

// Boost include(s).
#include <boost/test/unit_test.hpp>

namespace Acts {
namespace Cuda {
namespace Test {

BOOST_AUTO_TEST_SUITE(Utilities)
BOOST_AUTO_TEST_CASE(HostMemory) {
  // Matrix size helpers
  static constexpr std::size_t SIZE_X = 200;
  static constexpr std::size_t SIZE_Y = 120;
  static constexpr std::size_t SIZE_Z = 150;

  // Create and test a 1-dimensional array.
  Acts::Cuda::HostMemory<1, int> array({SIZE_X});
  BOOST_TEST_REQUIRE(array.size().at(0) == SIZE_X);
  BOOST_TEST_REQUIRE(array.totalSize() == SIZE_X);
  for(std::size_t i = 0; i < SIZE_X; ++i) {
    array.set({i}, 0);
  }
  array.set({0}, 314 );
  array.set({23}, 22 );
  BOOST_TEST_REQUIRE(array.get({0}) == 314);
  BOOST_TEST_REQUIRE(array.get({23}) == 22);
  BOOST_TEST_REQUIRE(array.get({100}) == 0);

  // Create and test a 2-dimensional matrix.
  Acts::Cuda::HostMemory<2, int> matrix2D({SIZE_X, SIZE_Y});
  BOOST_TEST_REQUIRE(matrix2D.size().at(0) == SIZE_X);
  BOOST_TEST_REQUIRE(matrix2D.size().at(1) == SIZE_Y);
  BOOST_TEST_REQUIRE(matrix2D.totalSize() == SIZE_X * SIZE_Y);
  for(std::size_t i = 0; i < SIZE_X; ++i) {
    for(std::size_t j = 0; j < SIZE_Y; ++j) {
      matrix2D.set({i, j}, 0);
    }
  }
  matrix2D.set({12, 44}, 224);
  matrix2D.set({47, 110}, 335);
  matrix2D.set({47}, 115);
  BOOST_TEST_REQUIRE(matrix2D.get({12, 44}) == 224);
  BOOST_TEST_REQUIRE(matrix2D.get({47, 110}) == 335);
  BOOST_TEST_REQUIRE(matrix2D.get({47, 0}) == 115);
  BOOST_TEST_REQUIRE(matrix2D.get({15, 40}) == 0);

  // Create and test a 3-dimensional matrix.
  Acts::Cuda::HostMemory<3, int> matrix3D({SIZE_X, SIZE_Y, SIZE_Z});
  BOOST_TEST_REQUIRE(matrix3D.size().at(0) == SIZE_X);
  BOOST_TEST_REQUIRE(matrix3D.size().at(1) == SIZE_Y);
  BOOST_TEST_REQUIRE(matrix3D.size().at(2) == SIZE_Z);
  BOOST_TEST_REQUIRE(matrix3D.totalSize() == SIZE_X * SIZE_Y * SIZE_Z);
  for(std::size_t i = 0; i < SIZE_X; ++i) {
    for(std::size_t j = 0; j < SIZE_Y; ++j) {
      for(std::size_t k = 0; k < SIZE_Z; ++k) {
        matrix3D.set({i, j, k}, 0);
      }
    }
  }
  matrix3D.set({67, 22, 88}, 11246);
  matrix3D.set({77, 23, 72}, 335);
  matrix3D.set({77, 23}, 334);
  matrix3D.set({77}, 333);
  BOOST_TEST_REQUIRE(matrix3D.get({67, 22, 88}) == 11246);
  BOOST_TEST_REQUIRE(matrix3D.get({77, 23, 72}) == 335);
  BOOST_TEST_REQUIRE(matrix3D.get({77, 23, 0}) == 334);
  BOOST_TEST_REQUIRE(matrix3D.get({77, 0, 0}) == 333);
}
BOOST_AUTO_TEST_SUITE_END()

}  // namespace Test
}  // namespace Cuda
}  // namespace Acts
