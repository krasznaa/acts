// This file is part of the Acts project.
//
// Copyright (C) 2020 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

// CUDA Plugin include(s).
#include "Acts/Plugins/Cuda/Utilities/HostMemory.hpp"

// System include(s).
#include <cassert>

namespace {

template<std::size_t DIM, std::size_t INDEX>
struct ArraySizeImpl {
  static_assert(INDEX < DIM);
  static std::size_t size(const std::array<std::size_t, DIM>& s) {
    static constexpr std::size_t LOWER_INDEX = INDEX - 1;
    return ArraySizeImpl<DIM, LOWER_INDEX>::size(s) * std::get<INDEX>(s);
  }
};  // struct ArraySizeImpl

template<std::size_t DIM>
struct ArraySizeImpl<DIM, 0> {
  static std::size_t size(const std::array<std::size_t, DIM>& s) {
    return std::get<0>(s);
  }
};  // struct ArraySizeImpl

template<std::size_t DIM>
struct ArraySize {
  static std::size_t size(const std::array<std::size_t, DIM>& s) {
    static constexpr std::size_t START_INDEX = DIM - 1;
    return ArraySizeImpl<DIM, START_INDEX>::size(s);
  }
};  // struct ArraySize

template<std::size_t DIM, std::size_t INDEX>
struct ElementPositionImpl {
  static_assert(INDEX < DIM);
  static std::size_t get(const std::array<std::size_t, DIM>& s,
                         const std::array<std::size_t, DIM>& i) {
    static constexpr std::size_t LOWER_INDEX = INDEX - 1;
    assert(std::get<INDEX>(i) < std::get<INDEX>(s));
    return std::get<INDEX>(i) * ArraySizeImpl<DIM, LOWER_INDEX>::size(s) +
        ElementPositionImpl<DIM, LOWER_INDEX>::get(s, i);
  }
};  // struct ElementPositionImpl

template<std::size_t DIM>
struct ElementPositionImpl<DIM, 0> {
  static std::size_t get(const std::array<std::size_t, DIM>& s,
                         const std::array<std::size_t, DIM>& i) {
    assert(std::get<0>(i) < std::get<0>(s));
    return std::get<0>(i);
  }
};  // struct ElementPositionImpl

template<std::size_t DIM>
struct ElementPosition {
  static std::size_t get(const std::array<std::size_t, DIM>& s,
                         const std::array<std::size_t, DIM>& i) {
    static constexpr std::size_t START_INDEX = DIM - 1;
    return ElementPositionImpl<DIM, START_INDEX>::get(s, i);
  }
};  // struct ElementPosition

}  // private namespace

namespace Acts {
namespace Cuda {

template<std::size_t DIM, typename T>
HostMemory<DIM,T>::HostMemory(
    const std::array<std::size_t, DIMENSIONS>& size)
: m_size(size),
  m_array(make_host_array<Type>(::ArraySize<DIMENSIONS>::size(size))) {

}

template<std::size_t DIM, typename T>
std::size_t HostMemory<DIM,T>::totalSize() const {
  return ::ArraySize<DIMENSIONS>::size(m_size);
}

template<std::size_t DIM, typename T>
typename HostMemory<DIM,T>::pointer HostMemory<DIM,T>::data() {
  return m_array.get();
}

template<std::size_t DIM, typename T>
typename HostMemory<DIM,T>::const_pointer HostMemory<DIM,T>::data() const {
  return m_array.get();
}

template<std::size_t DIM, typename T>
typename HostMemory<DIM,T>::Type HostMemory<DIM,T>::get(
    const std::array<std::size_t, DIMENSIONS>& i) const {
  return m_array.get()[::ElementPosition<DIMENSIONS>::get(m_size, i)];
}

template<std::size_t DIM, typename T>
void HostMemory<DIM,T>::set(const std::array<std::size_t, DIMENSIONS>& i,
                              Type value) {
  m_array.get()[::ElementPosition<DIMENSIONS>::get(m_size, i)] = value;
  return;
}

}  // namespace Cuda
}  // namespace Acts

/// Helper macro for instantiating the template code for a given type
#define INST_HM_FOR_TYPE(TYPE) \
  template class Acts::Cuda::HostMemory<1, TYPE>; \
  template class Acts::Cuda::HostMemory<2, TYPE>; \
  template class Acts::Cuda::HostMemory<3, TYPE>

// Instantiate the templated functions for all primitive types.
INST_HM_FOR_TYPE(char);
INST_HM_FOR_TYPE(unsigned char);
INST_HM_FOR_TYPE(short);
INST_HM_FOR_TYPE(unsigned short);
INST_HM_FOR_TYPE(int);
INST_HM_FOR_TYPE(unsigned int);
INST_HM_FOR_TYPE(long);
INST_HM_FOR_TYPE(unsigned long);
INST_HM_FOR_TYPE(long long);
INST_HM_FOR_TYPE(unsigned long long);
INST_HM_FOR_TYPE(float);
INST_HM_FOR_TYPE(double);

// Clean up.
#undef INST_HM_FOR_TYPE
