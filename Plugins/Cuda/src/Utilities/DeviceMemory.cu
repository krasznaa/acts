// This file is part of the Acts project.
//
// Copyright (C) 2020 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

// CUDA Plugin include(s).
#include "Acts/Plugins/Cuda/Utilities/DeviceMemory.hpp"

namespace {

template<std::size_t DIM, std::size_t INDEX>
struct ArraySizeImpl {
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

}  // private namespace

namespace Acts {
namespace Cuda {

template<std::size_t DIM, typename T>
DeviceMemory<DIM,T>::DeviceMemory(
    const std::array<std::size_t, DIMENSIONS>& size)
: m_size(size),
  m_array(make_device_array<Type>(::ArraySize<DIMENSIONS>::size(size))) {

}

template<std::size_t DIM, typename T>
std::size_t DeviceMemory<DIM,T>::size() const {
  return ::ArraySize<DIMENSIONS>::size(m_size);
}

template<std::size_t DIM, typename T>
typename DeviceMemory<DIM,T>::pointer DeviceMemory<DIM,T>::data() {
  return m_array.get();
}

template<std::size_t DIM, typename T>
typename DeviceMemory<DIM,T>::const_pointer DeviceMemory<DIM,T>::data() const {
  return m_array.get();
}

}  // namespace Cuda
}  // namespace Acts

/// Helper macro for instantiating the template code for a given type
#define INST_DM_FOR_TYPE(TYPE) \
  template class Acts::Cuda::DeviceMemory<1, TYPE>; \
  template class Acts::Cuda::DeviceMemory<2, TYPE>; \
  template class Acts::Cuda::DeviceMemory<3, TYPE>

// Instantiate the templated functions for all primitive types.
INST_DM_FOR_TYPE(char);
INST_DM_FOR_TYPE(unsigned char);
INST_DM_FOR_TYPE(short);
INST_DM_FOR_TYPE(unsigned short);
INST_DM_FOR_TYPE(int);
INST_DM_FOR_TYPE(unsigned int);
INST_DM_FOR_TYPE(long);
INST_DM_FOR_TYPE(unsigned long);
INST_DM_FOR_TYPE(long long);
INST_DM_FOR_TYPE(unsigned long long);
INST_DM_FOR_TYPE(float);
INST_DM_FOR_TYPE(double);

// Clean up.
#undef INST_DM_FOR_TYPE
