/*
  example/example18.cpp -- Usage of structured numpy dtypes

  Copyright (c) 2016 Ivan Smirnov

  All rights reserved. Use of this source code is governed by a
  BSD-style license that can be found in the LICENSE file.
*/

#include "example.h"

#include <pybind11/numpy.h>
#include <cstdint>
#include <iostream>

namespace py = pybind11;

struct Struct {
    bool x;
    uint32_t y;
    float z;
};

struct PackedStruct {
    bool x;
    uint32_t y;
    float z;
} __attribute__((packed));

struct NestedStruct {
    Struct a;
    PackedStruct b;
} __attribute__((packed));

template <typename T>
py::array mkarray_via_buffer(size_t n) {
    return py::array(py::buffer_info(nullptr, sizeof(T),
                                     py::format_descriptor<T>::value(),
                                     1, { n }, { sizeof(T) }));
}

template <typename S>
py::array_t<S> create_recarray(size_t n) {
    auto arr = mkarray_via_buffer<S>(n);
    auto ptr = static_cast<S*>(arr.request().ptr);
    for (size_t i = 0; i < n; i++) {
        ptr[i].x = i % 2; ptr[i].y = (uint32_t) i; ptr[i].z = (float) i * 1.5f;
    }
    return arr;
}

py::array_t<NestedStruct> create_nested(size_t n) {
    auto arr = mkarray_via_buffer<NestedStruct>(n);
    auto ptr = static_cast<NestedStruct*>(arr.request().ptr);
    for (size_t i = 0; i < n; i++) {
        ptr[i].a.x = i % 2; ptr[i].a.y = (uint32_t) i; ptr[i].a.z = (float) i * 1.5f;
        ptr[i].b.x = (i + 1) % 2; ptr[i].b.y = (uint32_t) (i + 1); ptr[i].b.z = (float) (i + 1) * 1.5f;
    }
    return arr;

}

void init_ex18(py::module &m) {
    PYBIND11_DTYPE(Struct, x, y, z);
    PYBIND11_DTYPE(PackedStruct, x, y, z);
    PYBIND11_DTYPE(NestedStruct, a, b);

    m.def("create_rec_simple", &create_recarray<Struct>);
    m.def("create_rec_packed", &create_recarray<PackedStruct>);
    m.def("create_rec_nested", &create_nested);
}
