// This file is part of the Acts project.
//
// Copyright (C) 2020 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

// System/CUDA include(s).
#include <assert.h>

namespace {

template<std::size_t DIM, std::size_t INDEX, typename T>
struct CopyArrayImpl {
  static_assert(INDEX < DIM);
  __device__
  static void copy(const T in[DIM], T out[DIM]) {
    static constexpr std::size_t LOWER_INDEX = INDEX - 1;
    out[INDEX] = in[INDEX];
    CopyArrayImpl<DIM, LOWER_INDEX, T>::copy(in, out);
    return;
  }
};  // struct CopyArrayImpl

template<std::size_t DIM, typename T>
struct CopyArrayImpl<DIM, 0, T> {
  static_assert(DIM > 0);
  __device__
  static void copy(const T in[DIM], T out[DIM]) {
    out[0] = in[0];
    return;
  }
};  // struct CopyArrayImpl

template<std::size_t DIM, typename T>
struct CopyArray {
  static_assert(DIM > 0);
  __device__
  static void copy(const T in[DIM], T out[DIM]) {
    static constexpr std::size_t START_INDEX = DIM - 1;
    CopyArrayImpl<DIM, START_INDEX, T>::copy(in, out);
    return;
  }
};  // struct CopyArray

template<std::size_t DIM, std::size_t INDEX>
struct ArraySizeImpl {
  static_assert(INDEX < DIM);
  __device__
  static std::size_t size(const std::size_t s[DIM]) {
    static constexpr std::size_t LOWER_INDEX = INDEX - 1;
    return ArraySizeImpl<DIM, LOWER_INDEX>::size(s) * s[INDEX];
  }
};  // struct ArraySizeImpl

template<std::size_t DIM>
struct ArraySizeImpl<DIM, 0> {
  __device__
  static std::size_t size(const std::size_t s[DIM]) {
    return s[0];
  }
};  // struct ArraySizeImpl

template<std::size_t DIM>
struct ArraySize {
  static_assert(DIM > 0);
  __device__
  static std::size_t size(const std::size_t s[DIM]) {
    static constexpr std::size_t START_INDEX = DIM - 1;
    return ArraySizeImpl<DIM, START_INDEX>::size(s);
  }
};  // struct ArraySize

template<std::size_t DIM, std::size_t INDEX>
struct ElementPositionImpl {
  static_assert(INDEX < DIM);
  __device__
  static std::size_t get(const std::size_t s[DIM],
                         const std::size_t i[DIM]) {
    static constexpr std::size_t LOWER_INDEX = INDEX - 1;
    assert(i[INDEX] < s[INDEX]);
    return i[INDEX] * ArraySizeImpl<DIM, LOWER_INDEX>::size(s) +
        ElementPositionImpl<DIM, LOWER_INDEX>::get(s, i);
  }
};  // struct ElementPositionImpl

template<std::size_t DIM>
struct ElementPositionImpl<DIM, 0> {
  static_assert(DIM > 0);
  __device__
  static std::size_t get(const std::size_t s[DIM],
                         const std::size_t i[DIM]) {
    assert(i[0] < s[0]);
    return i[0];
  }
};  // struct ElementPositionImpl

template<std::size_t DIM>
struct ElementPosition {
  static_assert(DIM > 0);
  __device__
  static std::size_t get(const std::size_t s[DIM],
                         const std::size_t i[DIM]) {
    static constexpr std::size_t START_INDEX = DIM - 1;
    return ElementPositionImpl<DIM, START_INDEX>::get(s, i);
  }
};  // struct ElementPosition

} // private namespace

namespace Acts {
namespace Cuda {

template<std::size_t DIM, typename T>
__device__
DeviceMatrix<DIM, T>::DeviceMatrix(const std::size_t size[DIMENSIONS],
                                   const_pointer ptr)
: m_array(ptr), m_ncArray(nullptr) {
  ::CopyArray<DIMENSIONS, std::size_t>::copy(size, m_size);
}

template<std::size_t DIM, typename T>
__device__
DeviceMatrix<DIM, T>::DeviceMatrix(const std::size_t size[DIMENSIONS],
                                   pointer ptr)
: m_array(ptr), m_ncArray(ptr) {
  ::CopyArray<DIMENSIONS, std::size_t>::copy(size, m_size);
}

template<std::size_t DIM, typename T>
__device__
std::size_t DeviceMatrix<DIM, T>::totalSize() const {
  return ::ArraySize<DIMENSIONS>::size(m_size);
}

template<std::size_t DIM, typename T>
__device__
typename DeviceMatrix<DIM, T>::pointer DeviceMatrix<DIM, T>::data() {
  return m_ncArray;
}

template<std::size_t DIM, typename T>
__device__
typename DeviceMatrix<DIM, T>::const_pointer
DeviceMatrix<DIM, T>::data() const {
  return m_array;
}

template<std::size_t DIM, typename T>
__device__
const typename DeviceMatrix<DIM, T>::Type& DeviceMatrix<DIM, T>::get(
    std::size_t i[DIMENSIONS]) const {
  return m_array[::ElementPosition<DIMENSIONS>::get(m_size, i)];
}

template<std::size_t DIM, typename T>
__device__
typename DeviceMatrix<DIM, T>::Type& DeviceMatrix<DIM, T>::getNC(
    std::size_t i[DIMENSIONS]) {
  assert(m_ncArray == m_array);
  return m_ncArray[::ElementPosition<DIMENSIONS>::get(m_size, i)];
}

template<std::size_t DIM, typename T>
__device__
void DeviceMatrix<DIM, T>::set(std::size_t i[DIMENSIONS], const Type& value) {
  assert(m_ncArray == m_array);
  m_ncArray[::ElementPosition<DIMENSIONS>::get(m_size, i)] = value;
  return;
}

}  // namespace Cuda
}  // namespace Acts
