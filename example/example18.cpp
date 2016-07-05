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

struct SimpleStruct {
    bool x;
    uint32_t y;
    float z;
};

std::ostream& operator<<(std::ostream& os, const SimpleStruct& v) {
    return os << "s:" << v.x << "," << v.y << "," << v.z;
}

struct PackedStruct {
    bool x;
    uint32_t y;
    float z;
} __attribute__((packed));

std::ostream& operator<<(std::ostream& os, const PackedStruct& v) {
    return os << "p:" << v.x << "," << v.y << "," << v.z;
}

struct NestedStruct {
    SimpleStruct a;
    PackedStruct b;
} __attribute__((packed));

std::ostream& operator<<(std::ostream& os, const NestedStruct& v) {
    return os << "n:a=" << v.a << ";b=" << v.b;
}

struct PartialStruct {
    bool x;
    uint32_t y;
    float z;
    long dummy2;
};

struct PartialNestedStruct {
    long dummy1;
    PartialStruct a;
    long dummy2;
};

struct UnboundStruct { };

template <typename T>
py::array mkarray_via_buffer(size_t n) {
    return py::array(py::buffer_info(nullptr, sizeof(T),
                                     py::format_descriptor<T>::format(),
                                     1, { n }, { sizeof(T) }));
}

template <typename S>
py::array_t<S, 0> create_recarray(size_t n) {
    auto arr = mkarray_via_buffer<S>(n);
    auto ptr = static_cast<S*>(arr.request().ptr);
    for (size_t i = 0; i < n; i++) {
        ptr[i].x = i % 2; ptr[i].y = (uint32_t) i; ptr[i].z = (float) i * 1.5f;
    }
    return arr;
}

std::string get_format_unbound() {
    return py::format_descriptor<UnboundStruct>::format();
}

py::array_t<NestedStruct, 0> create_nested(size_t n) {
    auto arr = mkarray_via_buffer<NestedStruct>(n);
    auto ptr = static_cast<NestedStruct*>(arr.request().ptr);
    for (size_t i = 0; i < n; i++) {
        ptr[i].a.x = i % 2; ptr[i].a.y = (uint32_t) i; ptr[i].a.z = (float) i * 1.5f;
        ptr[i].b.x = (i + 1) % 2; ptr[i].b.y = (uint32_t) (i + 1); ptr[i].b.z = (float) (i + 1) * 1.5f;
    }
    return arr;
}

py::array_t<PartialNestedStruct, 0> create_partial_nested(size_t n) {
    auto arr = mkarray_via_buffer<PartialNestedStruct>(n);
    auto ptr = static_cast<PartialNestedStruct*>(arr.request().ptr);
    for (size_t i = 0; i < n; i++) {
        ptr[i].a.x = i % 2; ptr[i].a.y = (uint32_t) i; ptr[i].a.z = (float) i * 1.5f;
    }
    std::cout << "dtype! " << (std::string) ((py::object) arr.attr("dtype")).str() << "\n";
    return arr;
}

template <typename S>
void print_recarray(py::array_t<S> arr) {
    auto buf = arr.request();
    auto ptr = static_cast<S*>(buf.ptr);
    for (size_t i = 0; i < buf.size; i++)
        std::cout << ptr[i] << std::endl;
}

void print_format_descriptors() {
    std::cout << py::format_descriptor<SimpleStruct>::format() << std::endl;
    std::cout << py::format_descriptor<PackedStruct>::format() << std::endl;
    std::cout << py::format_descriptor<NestedStruct>::format() << std::endl;
    std::cout << py::format_descriptor<PartialStruct>::format() << std::endl;
    std::cout << py::format_descriptor<PartialNestedStruct>::format() << std::endl;
}

void print_dtypes() {
    auto to_str = [](py::object obj) {
        return (std::string) (py::str) ((py::object) obj.attr("__str__"))();
    };
    std::cout << to_str(py::dtype_of<SimpleStruct>()) << std::endl;
    std::cout << to_str(py::dtype_of<PackedStruct>()) << std::endl;
    std::cout << to_str(py::dtype_of<NestedStruct>()) << std::endl;
    std::cout << to_str(py::dtype_of<PartialStruct>()) << std::endl;
    std::cout << to_str(py::dtype_of<PartialNestedStruct>()) << std::endl;
}

void init_ex18(py::module &m) {
    PYBIND11_NUMPY_DTYPE(SimpleStruct, x, y, z);
    PYBIND11_NUMPY_DTYPE(PackedStruct, x, y, z);
    PYBIND11_NUMPY_DTYPE(NestedStruct, a, b);
    PYBIND11_NUMPY_DTYPE(PartialStruct, x, y, z);
    PYBIND11_NUMPY_DTYPE(PartialNestedStruct, a);

    m.def("create_rec_simple", &create_recarray<SimpleStruct>);
    m.def("create_rec_packed", &create_recarray<PackedStruct>);
    m.def("create_rec_nested", &create_nested);
    m.def("create_rec_partial", &create_recarray<PartialStruct>);
    m.def("create_rec_partial_nested", &create_partial_nested);
    m.def("print_format_descriptors", &print_format_descriptors);
    m.def("print_rec_simple", &print_recarray<SimpleStruct>);
    m.def("print_rec_packed", &print_recarray<PackedStruct>);
    m.def("print_rec_nested", &print_recarray<NestedStruct>);
    m.def("print_dtypes", &print_dtypes);
    m.def("get_format_unbound", &get_format_unbound);
    m.def("spf", [](py::object o) { return py::array::strip_padding_fields(o); });
}
