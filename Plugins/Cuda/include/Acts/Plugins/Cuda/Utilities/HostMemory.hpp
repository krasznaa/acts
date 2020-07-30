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
#include <array>
#include <cstddef>

namespace Acts {
namespace Cuda {

/// Helper type for handling 1, 2 and 3-dimensional arrays/matrices in device
/// memory
template<std::size_t DIM, typename T>
class HostMemory {

  /// Make sure that we have at least a single dimension
  static_assert(DIM > 0);

 public:
  /// The dimensionality of this object
  static constexpr std::size_t DIMENSIONS = DIM;
  /// The type of the underlying primitive type
  using Type = T;
  /// Non-constant pointer to the memory block
  using pointer = Type*;
  /// Constant pointer to the memory block
  using const_pointer = const Type*;

  /// Create the array/matrix in device memory
  HostMemory(const std::array<std::size_t, DIMENSIONS>& size);

  /// Get the array describing the size of the memory block
  const std::array<std::size_t, DIMENSIONS>& size() const { return m_size; }
  /// Get the total size of the underlying memory block
  std::size_t totalSize() const;

  /// Get a (non-constant) pointer to the underlying memory block
  pointer data();
  /// Get a (constant) pointer to the underlying memory block
  const_pointer data() const;

  /// Get one element of the array/matrix
  Type get(const std::array<std::size_t, DIMENSIONS>& i) const;
  /// Set one element of the array/matrix
  void set(const std::array<std::size_t, DIMENSIONS>& i, Type value);

 private:
  /// The size of the array/matrix in memory
  std::array<std::size_t, DIMENSIONS> m_size;
  /// Smart pointer managing the device memory
  host_array<Type> m_array;

};  // class HostMemory

}  // namespace Cuda
}  // namespace Acts
