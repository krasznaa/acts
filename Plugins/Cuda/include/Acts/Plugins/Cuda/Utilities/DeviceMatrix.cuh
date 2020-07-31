// This file is part of the Acts project.
//
// Copyright (C) 2020 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

namespace Acts {
namespace Cuda {

/// Helper type for handling 1, 2 and 3-dimensional matrices on the device
template<unsigned int DIM, typename T>
class DeviceMatrix {

  /// Make sure that we have at least a single dimension for the matrix
  static_assert(DIM > 0);

 public:
  /// The dimensionality of this object
  static constexpr unsigned int DIMENSIONS = DIM;
  /// The type of the underlying primitive type
  using Type = T;
  /// Non-constant pointer to the memory block
  using pointer = Type*;
  /// Constant pointer to the memory block
  using const_pointer = const Type*;

  /// Create the helper object around a constant array
  __device__
  DeviceMatrix(const unsigned int size[DIMENSIONS], const_pointer ptr);
  /// Create the helper object around a non-constant array
  __device__
  DeviceMatrix(const unsigned int size[DIMENSIONS], pointer ptr);

  /// Get the array describing the size of the memory block
  __device__
  const unsigned int* size() const { return m_size; }
  /// Get the total size of the underlying memory block
  __device__
  unsigned int totalSize() const;

  /// Get a (non-constant) pointer to the underlying memory block
  __device__
  pointer data();
  /// Get a (constant) pointer to the underlying memory block
  __device__
  const_pointer data() const;

  /// Get one element of the matrix
  __device__
  Type get(unsigned int i[DIMENSIONS]) const;
  /// Set one element of the matrix
  __device__
  void set(unsigned int i[DIMENSIONS], Type value);

 private:
  /// The size of the matrix in memory
  unsigned int m_size[DIMENSIONS] = {0};
  /// Constant pointer to the memory block
  const_pointer m_array;
  /// Non-constant pointer to the memory block
  pointer m_ncArray;

};  // class DeviceMatrix

}  // namespace Cuda
}  // namespace Acts

// Include the implementation.
#include "DeviceMatrix.ipp"
