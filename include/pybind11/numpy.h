/*
    pybind11/numpy.h: Basic NumPy support, vectorize() wrapper

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#pragma once

#include "pybind11.h"
#include "complex.h"
#include <numeric>
#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <initializer_list>

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable: 4127) // warning C4127: Conditional expression is constant
#endif

NAMESPACE_BEGIN(pybind11)
namespace detail {
template <typename type, typename SFINAE = void> struct npy_format_descriptor { };
template <typename type> struct is_pod_struct;

struct PyArrayDescr_Proxy {
    PyObject_HEAD
    PyObject *typeobj;
    char kind;
    char type;
    char byteorder;
    char flags;
    int type_num;
    int elsize;
    int alignment;
    char *subarray;
    PyObject *fields;
    PyObject *names;
};

struct PyArray_Proxy {
    PyObject_HEAD
    char *data;
    int nd;
    ssize_t *dimensions;
    ssize_t *strides;
    PyObject *base;
    PyObject *descr;
    int flags;
};

struct npy_api {
    enum constants {
        NPY_C_CONTIGUOUS_ = 0x0001,
        NPY_F_CONTIGUOUS_ = 0x0002,
        NPY_ENSURE_ARRAY_ = 0x0040,
        NPY_ARRAY_ALIGNED_ = 0x0100,
        NPY_ARRAY_WRITEABLE_ = 0x0400,
        NPY_BOOL_ = 0,
        NPY_BYTE_, NPY_UBYTE_,
        NPY_SHORT_, NPY_USHORT_,
        NPY_INT_, NPY_UINT_,
        NPY_LONG_, NPY_ULONG_,
        NPY_LONGLONG_, NPY_ULONGLONG_,
        NPY_FLOAT_, NPY_DOUBLE_, NPY_LONGDOUBLE_,
        NPY_CFLOAT_, NPY_CDOUBLE_, NPY_CLONGDOUBLE_,
        NPY_OBJECT_ = 17,
        NPY_STRING_, NPY_UNICODE_, NPY_VOID_
    };

    static npy_api& get() {
        static npy_api api = lookup();
        return api;
    }

    bool PyArray_Check_(PyObject *obj) const {
        return (bool) PyObject_TypeCheck(obj, PyArray_Type_);
    }
    bool PyArrayDescr_Check_(PyObject *obj) const {
        return (bool) PyObject_TypeCheck(obj, PyArrayDescr_Type_);
    }

    PyObject *(*PyArray_DescrFromType_)(int);
    PyObject *(*PyArray_NewFromDescr_)
        (PyTypeObject *, PyObject *, int, Py_intptr_t *,
         Py_intptr_t *, void *, int, PyObject *);
    PyObject *(*PyArray_DescrNewFromType_)(int);
    PyObject *(*PyArray_NewCopy_)(PyObject *, int);
    PyTypeObject *PyArray_Type_;
    PyTypeObject *PyArrayDescr_Type_;
    PyObject *(*PyArray_FromAny_) (PyObject *, PyObject *, int, int, int, PyObject *);
    int (*PyArray_CopyInto_) (PyObject *, PyObject *);
    PyObject *(*PyArray_FromArray_) (PyObject *, PyObject *, int);
    int (*PyArray_DescrConverter_) (PyObject *, PyObject **);
    bool (*PyArray_EquivTypes_) (PyObject *, PyObject *);
    bool (*PyArray_CanCastArrayTo_) (PyObject *, PyObject *, int);
    PyObject *(*PyArray_NewLikeArray_) (PyObject *, int, PyObject *, int);
    int (*PyArray_GetArrayParamsFromObject_)(PyObject *, PyObject *, char, PyObject **, int *,
                                             Py_ssize_t *, PyObject **, PyObject *);
private:
    enum functions {
        API_PyArray_Type = 2,
        API_PyArrayDescr_Type = 3,
        API_PyArray_DescrFromType = 45,
        API_PyArray_FromAny = 69,
        API_PyArray_CopyInto = 82,
        API_PyArray_NewCopy = 85,
        API_PyArray_NewFromDescr = 94,
        API_PyArray_DescrNewFromType = 9,
        API_PyArray_FromArray = 109,
        API_PyArray_DescrConverter = 174,
        API_PyArray_EquivTypes = 182,
        API_PyArray_CanCastArrayTo = 274,
        API_PyArray_NewLikeArray = 277,
        API_PyArray_GetArrayParamsFromObject = 278,
    };

    static npy_api lookup() {
        module m = module::import("numpy.core.multiarray");
        object c = (object) m.attr("_ARRAY_API");
#if PY_MAJOR_VERSION >= 3
        void **api_ptr = (void **) (c ? PyCapsule_GetPointer(c.ptr(), NULL) : nullptr);
#else
        void **api_ptr = (void **) (c ? PyCObject_AsVoidPtr(c.ptr()) : nullptr);
#endif
        npy_api api;
#define DECL_NPY_API(Func) api.Func##_ = (decltype(api.Func##_)) api_ptr[API_##Func];
        DECL_NPY_API(PyArray_Type);
        DECL_NPY_API(PyArrayDescr_Type);
        DECL_NPY_API(PyArray_DescrFromType);
        DECL_NPY_API(PyArray_FromAny);
        DECL_NPY_API(PyArray_CopyInto);
        DECL_NPY_API(PyArray_FromArray);
        DECL_NPY_API(PyArray_NewCopy);
        DECL_NPY_API(PyArray_NewFromDescr);
        DECL_NPY_API(PyArray_DescrNewFromType);
        DECL_NPY_API(PyArray_DescrConverter);
        DECL_NPY_API(PyArray_EquivTypes);
        DECL_NPY_API(PyArray_CanCastArrayTo);
        DECL_NPY_API(PyArray_NewLikeArray);
        DECL_NPY_API(PyArray_GetArrayParamsFromObject);
#undef DECL_NPY_API
        return api;
    }
};
}

#define PyArray_GET(ptr, attr) (reinterpret_cast<::pybind11::detail::PyArray_Proxy*>(ptr)->attr)
#define PyArrayDescr_GET(ptr, attr) (reinterpret_cast<::pybind11::detail::PyArrayDescr_Proxy*>(ptr)->attr)

class dtype : public object {
public:
    PYBIND11_OBJECT_DEFAULT(dtype, object, detail::npy_api::get().PyArrayDescr_Check_);

    dtype(const buffer_info &info) {
        dtype descr(_dtype_from_pep3118()(PYBIND11_STR_TYPE(info.format)));
        m_ptr = descr.strip_padding().release().ptr();
    }

    dtype(std::string format) {
        m_ptr = from_args(pybind11::str(format)).release().ptr();
    }

    dtype(const char *format) : dtype(std::string(format)) { }

    dtype(list names, list formats, list offsets, size_t itemsize) {
        dict args;
        args["names"] = names;
        args["formats"] = formats;
        args["offsets"] = offsets;
        args["itemsize"] = pybind11::int_(itemsize);
        m_ptr = from_args(args).release().ptr();
    }

    static dtype from_args(object args) {
        // This is essentially the same as calling np.dtype() constructor in Python
        PyObject *ptr = nullptr;
        if (!detail::npy_api::get().PyArray_DescrConverter_(args.release().ptr(), &ptr) || !ptr)
            pybind11_fail("NumPy: failed to create structured dtype");
        return object(ptr, false);
    }

    template <typename T> static dtype of() {
        return detail::npy_format_descriptor<typename std::remove_cv<T>::type>::dtype();
    }

    size_t itemsize() const {
        return (size_t) PyArrayDescr_GET(m_ptr, elsize);
    }

    bool has_fields() const {
        return PyArrayDescr_GET(m_ptr, names) != nullptr;
    }

    char kind() const {
        return PyArrayDescr_GET(m_ptr, kind);
    }

private:
    static object _dtype_from_pep3118() {
        static PyObject *obj = module::import("numpy.core._internal")
            .attr("_dtype_from_pep3118").cast<object>().release().ptr();
        return object(obj, true);
    }

    dtype strip_padding() {
        // Recursively strip all void fields with empty names that are generated for
        // padding fields (as of NumPy v1.11).
        if (!has_fields())
            return *this;

        struct field_descr { PYBIND11_STR_TYPE name; object format; pybind11::int_ offset; };
        std::vector<field_descr> field_descriptors;

        auto fields = attr("fields").cast<object>();
        auto items = fields.attr("items").cast<object>();
        for (auto field : items()) {
            auto spec = object(field, true).cast<tuple>();
            auto name = spec[0].cast<pybind11::str>();
            auto format = spec[1].cast<tuple>()[0].cast<dtype>();
            auto offset = spec[1].cast<tuple>()[1].cast<pybind11::int_>();
            if (!len(name) && format.kind() == 'V')
                continue;
            field_descriptors.push_back({(PYBIND11_STR_TYPE) name, format.strip_padding(), offset});
        }

        std::sort(field_descriptors.begin(), field_descriptors.end(),
                  [](const field_descr& a, const field_descr& b) {
                      return a.offset.cast<int>() < b.offset.cast<int>();
                  });

        list names, formats, offsets;
        for (auto& descr : field_descriptors) {
            names.append(descr.name);
            formats.append(descr.format);
            offsets.append(descr.offset);
        }
        return dtype(names, formats, offsets, itemsize());
    }
};

namespace detail {
template<typename T> inline PyObject* get_dtype_ptr() {
    return dtype::of<T>().release().inc_ref().ptr(); }

template<> inline PyObject* get_dtype_ptr<void>() {
    return nullptr;
}
}

enum class casting : int { no, equiv, safe, same_kind, unsafe };

inline const char* to_string(casting policy) {
    static const char *policies[] = { "no", "equiv", "safe", "same_kind", "unsafe" };
    return policies[static_cast<int>(policy)];
}

struct array_flags {
    enum {
        c_style = detail::npy_api::NPY_C_CONTIGUOUS_,
        f_style = detail::npy_api::NPY_F_CONTIGUOUS_,
        aligned = detail::npy_api::NPY_ARRAY_ALIGNED_,
        writeable = detail::npy_api::NPY_ARRAY_WRITEABLE_,
        in_flags = c_style | aligned,
        out_flags = c_style | aligned | writeable,
        default_flags = out_flags
    };
};

template<typename T, int flags> class array_t;

template<int flags>
class generic_array : public buffer {
public:
    PYBIND11_OBJECT_CVT(
        generic_array,
        buffer,
        check_array,
        m_ptr = ensure_array(m_ptr)
    )

    using data_ptr_t = typename std::conditional<!!(flags & array_flags::writeable), void *, const void *>::type;

    enum {
        c_style = array_flags::c_style,
        f_style = array_flags::f_style,
        aligned = array_flags::aligned,
        writeable = array_flags::writeable,
        in_flags = array_flags::in_flags,
        out_flags = array_flags::out_flags,
        default_flags = array_flags::default_flags
    };
    using casting = pybind11::casting;

    static_assert(!(flags & c_style) || !(flags & f_style),
                  "invalid flags: cannot combine c_style and f_style");

    generic_array() : buffer() { }

    generic_array(const pybind11::dtype& dt, const std::vector<size_t>& shape,
                  const std::vector<size_t>& strides, void *ptr = nullptr) {
        auto& api = detail::npy_api::get();
        auto ndim = shape.size();
        if (shape.size() != strides.size())
            pybind11_fail("NumPy: shape ndim doesn't match strides ndim");
        auto descr = dt;
        object tmp(api.PyArray_NewFromDescr_(
            api.PyArray_Type_, descr.release().ptr(), (int) ndim, (Py_intptr_t *) shape.data(),
            (Py_intptr_t *) strides.data(), ptr, 0, nullptr), false);
        if (!tmp)
            throw error_already_set();
        if (ptr)
            tmp = object(api.PyArray_NewCopy_(tmp.ptr(), -1 /* any order */), false);
        m_ptr = tmp.release().ptr();
    }

    generic_array(const pybind11::dtype& dt, const std::vector<size_t>& shape, void *ptr = nullptr)
    : generic_array(dt, shape, c_strides(shape, dt.itemsize()), ptr) { }

    generic_array(const pybind11::dtype& dt, size_t count, void *ptr = nullptr)
    : generic_array(dt, std::vector<size_t> { count }, ptr) { }

    template<typename T> generic_array(const std::vector<size_t>& shape,
                                       const std::vector<size_t>& strides, T* ptr)
    : generic_array(pybind11::dtype::of<T>(), shape, strides, (void *) ptr) { }

    template<typename T> generic_array(const std::vector<size_t>& shape, T* ptr)
    : generic_array(shape, c_strides(shape, sizeof(T)), ptr) { }

    template<typename T> generic_array(size_t count, T* ptr)
    : generic_array(std::vector<size_t> { count }, ptr) { }

    generic_array(const buffer_info &info)
    : generic_array(pybind11::dtype(info), info.shape, info.strides, info.ptr) { }

    generic_array<flags> astype(const pybind11::dtype& newtype, casting policy = casting::same_kind) const;
    template<typename T> array_t<T, flags> astype(casting policy = casting::same_kind) const;

    pybind11::dtype dtype() const {
        return object(PyArray_GET(m_ptr, descr), true);
    }

    size_t ndim() const {
        return (size_t) PyArray_GET(m_ptr, nd);
    }

    const size_t* shape() const {
        static_assert(sizeof(size_t) == sizeof(Py_intptr_t), "size_t != Py_intptr_t");
        return reinterpret_cast<const size_t *>(PyArray_GET(m_ptr, dimensions));
    }

    size_t shape(size_t axis) const {
        if (axis >= ndim())
            pybind11_fail("NumPy: attempted to index shape beyond ndim");
        return shape()[axis];
    }

    const size_t* strides() const {
        static_assert(sizeof(size_t) == sizeof(Py_intptr_t), "size_t != Py_intptr_t");
        return reinterpret_cast<const size_t *>(PyArray_GET(m_ptr, strides));
    }

    size_t strides(size_t axis) const {
        if (axis >= ndim())
            pybind11_fail("NumPy: attempted to index strides beyond ndim");
        return strides()[axis];
    }

    data_ptr_t data() const {
        return reinterpret_cast<data_ptr_t>(PyArray_GET(m_ptr, data));
    }

protected:
    template <typename T, typename SFINAE> friend struct detail::npy_format_descriptor;

    static std::vector<size_t> c_strides(const std::vector<size_t>& shape, size_t itemsize) {
        auto ndim = shape.size();
        std::vector<size_t> strides(ndim);
        if (ndim) {
            std::fill(strides.begin(), strides.end(), itemsize);
            for (size_t i = 0; i < ndim - 1; i++)
                for (size_t j = 0; j < ndim - 1 - i; j++)
                    strides[j] *= shape[ndim - 1 - i];
        }
        return strides;
    }

    static bool check_array(PyObject *ptr) {
        return ptr && detail::npy_api::get().PyArray_Check_(ptr);
    }

    template<typename T = void>
    static PyObject* ensure_array(PyObject *ptr) {
        auto& api = detail::npy_api::get();
        int arr_flags = detail::npy_api::NPY_ENSURE_ARRAY_ | flags;
        auto descr = detail::get_dtype_ptr<T>();
        auto result = api.PyArray_FromAny_(ptr, descr, 0, 0, arr_flags, nullptr);
        if (!result)
            throw error_already_set();
        Py_DECREF(ptr);
        return result;
    }
};

class array_in : public generic_array<array_flags::in_flags> {
    using generic_array::generic_array;
};
class array_out : public generic_array<array_flags::out_flags> {
    using generic_array::generic_array;
};
class array : public generic_array<array_flags::default_flags> {
    using generic_array::generic_array;
    using in = array_in;
    using out = array_out;
};

template<typename T, int flags = array_flags::default_flags>
class array_t : public generic_array<flags> {
    using base_t = generic_array<flags>;

public:
    PYBIND11_OBJECT_CVT(
        array_t,
        base_t,
        base_t::check_array,
        this->m_ptr = base_t::template ensure_array<T>(this->m_ptr)
    )

    using in = array_t<T, array_flags::in_flags>;
    using out = array_t<T, array_flags::out_flags>;

    using data_ptr_t = typename std::conditional<!!(flags & array_flags::writeable), T *, const T *>::type;

    array_t() : base_t() { }

    array_t(const buffer_info& info) : base_t(info) { }

    array_t(const std::vector<size_t>& shape, const std::vector<size_t>& strides, T* ptr = nullptr)
    : base_t(shape, strides, ptr) { }

    array_t(const std::vector<size_t>& shape, T* ptr = nullptr)
    : base_t(shape, ptr) { }

    array_t(size_t count, T* ptr = nullptr)
    : base_t(count, ptr) { }

    data_ptr_t data() {
        return reinterpret_cast<data_ptr_t>(PyArray_GET(this->m_ptr, data));
    }
};
template<typename T> using array_in_t = typename array_t<T>::in;
template<typename T> using array_out_t = typename array_t<T>::out;

namespace detail {
inline PyObject* cast_array(PyObject* src, const dtype& newtype, casting policy) {
    auto& api = npy_api::get();
    if (!api.PyArray_CanCastArrayTo_(src, newtype.ptr(), static_cast<int>(policy)))
        pybind11_fail(std::string("NumPy: unable to cast array to type `") +
                      newtype.str().cast<std::string>() + "` using policy `" +
                      to_string(policy) + "`");
    auto arr = handle(api.PyArray_NewLikeArray_(src, /* order = */ -1, newtype.ptr(), /* subok = */ 1));
    if (!arr)
        throw error_already_set();
    if (api.PyArray_CopyInto_(arr.ptr(), src) < 0 || !arr) {
        arr.dec_ref();
        throw error_already_set();
    }
    return arr.ptr();
}
}

template<int flags>
generic_array<flags> generic_array<flags>::astype(const pybind11::dtype& newtype, casting policy) const {
    return {detail::cast_array(m_ptr, newtype, policy), false};
}

template<int flags> template<typename T>
array_t<T, flags> generic_array<flags>::astype(casting policy) const {
    return {detail::cast_array(m_ptr, pybind11::dtype::of<T>(), policy), false};
}

template<int flags> struct detail::load_options<generic_array<flags>> {
    enum { propagate_errors = 1 };
};
template<typename T, int flags> struct detail::load_options<array_t<T, flags>> {
    enum { propagate_errors = 1 };
};

template <typename T>
struct format_descriptor<T, typename std::enable_if<detail::is_pod_struct<T>::value>::type> {
    static std::string format() {
        return detail::npy_format_descriptor<typename std::remove_cv<T>::type>::format();
    }
};

template <size_t N> struct format_descriptor<char[N]> {
    static std::string format() { return std::to_string(N) + "s"; }
};
template <size_t N> struct format_descriptor<std::array<char, N>> {
    static std::string format() { return std::to_string(N) + "s"; }
};

NAMESPACE_BEGIN(detail)
template <typename T> struct is_std_array : std::false_type { };
template <typename T, size_t N> struct is_std_array<std::array<T, N>> : std::true_type { };

template <typename T>
struct is_pod_struct {
    enum { value = std::is_pod<T>::value && // offsetof only works correctly for POD types
           !std::is_reference<T>::value &&
           !std::is_array<T>::value &&
           !is_std_array<T>::value &&
           !std::is_integral<T>::value &&
           !std::is_same<typename std::remove_cv<T>::type, float>::value &&
           !std::is_same<typename std::remove_cv<T>::type, double>::value &&
           !std::is_same<typename std::remove_cv<T>::type, bool>::value &&
           !std::is_same<typename std::remove_cv<T>::type, std::complex<float>>::value &&
           !std::is_same<typename std::remove_cv<T>::type, std::complex<double>>::value };
};

template <typename T> struct npy_format_descriptor<T, typename std::enable_if<std::is_integral<T>::value>::type> {
private:
    constexpr static const int values[8] = {
        npy_api::NPY_BYTE_, npy_api::NPY_UBYTE_, npy_api::NPY_SHORT_,    npy_api::NPY_USHORT_,
        npy_api::NPY_INT_,  npy_api::NPY_UINT_,  npy_api::NPY_LONGLONG_, npy_api::NPY_ULONGLONG_ };
public:
    enum { value = values[detail::log2(sizeof(T)) * 2 + (std::is_unsigned<T>::value ? 1 : 0)] };
    static pybind11::dtype dtype() {
        if (auto ptr = npy_api::get().PyArray_DescrFromType_(value))
            return object(ptr, true);
        pybind11_fail("Unsupported buffer format!");
    }
    template <typename T2 = T, typename std::enable_if<std::is_signed<T2>::value, int>::type = 0>
    static PYBIND11_DESCR name() { return _("int") + _<sizeof(T)*8>(); }
    template <typename T2 = T, typename std::enable_if<!std::is_signed<T2>::value, int>::type = 0>
    static PYBIND11_DESCR name() { return _("uint") + _<sizeof(T)*8>(); }
};
template <typename T> constexpr const int npy_format_descriptor<
    T, typename std::enable_if<std::is_integral<T>::value>::type>::values[8];

#define DECL_FMT(Type, NumPyName, Name) template<> struct npy_format_descriptor<Type> { \
    enum { value = npy_api::NumPyName }; \
    static pybind11::dtype dtype() { \
        if (auto ptr = npy_api::get().PyArray_DescrFromType_(value)) \
            return object(ptr, true); \
        pybind11_fail("Unsupported buffer format!"); \
    } \
    static PYBIND11_DESCR name() { return _(Name); } }
DECL_FMT(float, NPY_FLOAT_, "float32");
DECL_FMT(double, NPY_DOUBLE_, "float64");
DECL_FMT(bool, NPY_BOOL_, "bool");
DECL_FMT(std::complex<float>, NPY_CFLOAT_, "complex64");
DECL_FMT(std::complex<double>, NPY_CDOUBLE_, "complex128");
#undef DECL_FMT

#define DECL_CHAR_FMT \
    static PYBIND11_DESCR name() { return _("S") + _<N>(); } \
    static pybind11::dtype dtype() { return std::string("S") + std::to_string(N); }
template <size_t N> struct npy_format_descriptor<char[N]> { DECL_CHAR_FMT };
template <size_t N> struct npy_format_descriptor<std::array<char, N>> { DECL_CHAR_FMT };
#undef DECL_CHAR_FMT

struct field_descriptor {
    const char *name;
    size_t offset;
    size_t size;
    std::string format;
    dtype descr;
};

template <typename T>
struct npy_format_descriptor<T, typename std::enable_if<is_pod_struct<T>::value>::type> {
    static PYBIND11_DESCR name() { return _("struct"); }

    static pybind11::dtype dtype() {
        if (!dtype_ptr)
            pybind11_fail("NumPy: unsupported buffer format!");
        return object(dtype_ptr, true);
    }

    static std::string format() {
        if (!dtype_ptr)
            pybind11_fail("NumPy: unsupported buffer format!");
        return format_str;
    }

    static void register_dtype(std::initializer_list<field_descriptor> fields) {
        list names, formats, offsets;
        for (auto field : fields) {
            if (!field.descr)
                pybind11_fail("NumPy: unsupported field dtype");
            names.append(PYBIND11_STR_TYPE(field.name));
            formats.append(field.descr);
            offsets.append(pybind11::int_(field.offset));
        }
        dtype_ptr = pybind11::dtype(names, formats, offsets, sizeof(T)).release().ptr();

        // There is an existing bug in NumPy (as of v1.11): trailing bytes are
        // not encoded explicitly into the format string. This will supposedly
        // get fixed in v1.12; for further details, see these:
        // - https://github.com/numpy/numpy/issues/7797
        // - https://github.com/numpy/numpy/pull/7798
        // Because of this, we won't use numpy's logic to generate buffer format
        // strings and will just do it ourselves.
        std::vector<field_descriptor> ordered_fields(fields);
        std::sort(ordered_fields.begin(), ordered_fields.end(),
                  [](const field_descriptor& a, const field_descriptor &b) {
                      return a.offset < b.offset;
                  });
        size_t offset = 0;
        std::ostringstream oss;
        oss << "T{";
        for (auto& field : ordered_fields) {
            if (field.offset > offset)
                oss << (field.offset - offset) << 'x';
            // note that '=' is required to cover the case of unaligned fields
            oss << '=' << field.format << ':' << field.name << ':';
            offset = field.offset + field.size;
        }
        if (sizeof(T) > offset)
            oss << (sizeof(T) - offset) << 'x';
        oss << '}';
        format_str = oss.str();

        // Sanity check: verify that NumPy properly parses our buffer format string
        auto& api = npy_api::get();
        auto arr =  array(buffer_info(nullptr, sizeof(T), format(), 1));
        if (!api.PyArray_EquivTypes_(dtype_ptr, arr.dtype().ptr()))
            pybind11_fail("NumPy: invalid buffer descriptor!");
    }

private:
    static std::string format_str;
    static PyObject* dtype_ptr;
};

template <typename T>
std::string npy_format_descriptor<T, typename std::enable_if<is_pod_struct<T>::value>::type>::format_str;
template <typename T>
PyObject* npy_format_descriptor<T, typename std::enable_if<is_pod_struct<T>::value>::type>::dtype_ptr = nullptr;

// Extract name, offset and format descriptor for a struct field
#define PYBIND11_FIELD_DESCRIPTOR(Type, Field) \
    ::pybind11::detail::field_descriptor { \
        #Field, offsetof(Type, Field), sizeof(decltype(static_cast<Type*>(0)->Field)), \
        ::pybind11::format_descriptor<decltype(static_cast<Type*>(0)->Field)>::format(), \
        ::pybind11::detail::npy_format_descriptor<decltype(static_cast<Type*>(0)->Field)>::dtype() \
    }

// The main idea of this macro is borrowed from https://github.com/swansontec/map-macro
// (C) William Swanson, Paul Fultz
#define PYBIND11_EVAL0(...) __VA_ARGS__
#define PYBIND11_EVAL1(...) PYBIND11_EVAL0 (PYBIND11_EVAL0 (PYBIND11_EVAL0 (__VA_ARGS__)))
#define PYBIND11_EVAL2(...) PYBIND11_EVAL1 (PYBIND11_EVAL1 (PYBIND11_EVAL1 (__VA_ARGS__)))
#define PYBIND11_EVAL3(...) PYBIND11_EVAL2 (PYBIND11_EVAL2 (PYBIND11_EVAL2 (__VA_ARGS__)))
#define PYBIND11_EVAL4(...) PYBIND11_EVAL3 (PYBIND11_EVAL3 (PYBIND11_EVAL3 (__VA_ARGS__)))
#define PYBIND11_EVAL(...)  PYBIND11_EVAL4 (PYBIND11_EVAL4 (PYBIND11_EVAL4 (__VA_ARGS__)))
#define PYBIND11_MAP_END(...)
#define PYBIND11_MAP_OUT
#define PYBIND11_MAP_COMMA ,
#define PYBIND11_MAP_GET_END() 0, PYBIND11_MAP_END
#define PYBIND11_MAP_NEXT0(test, next, ...) next PYBIND11_MAP_OUT
#define PYBIND11_MAP_NEXT1(test, next) PYBIND11_MAP_NEXT0 (test, next, 0)
#define PYBIND11_MAP_NEXT(test, next)  PYBIND11_MAP_NEXT1 (PYBIND11_MAP_GET_END test, next)
#ifdef _MSC_VER // MSVC is not as eager to expand macros, hence this workaround
#define PYBIND11_MAP_LIST_NEXT1(test, next) \
    PYBIND11_EVAL0 (PYBIND11_MAP_NEXT0 (test, PYBIND11_MAP_COMMA next, 0))
#else
#define PYBIND11_MAP_LIST_NEXT1(test, next) \
    PYBIND11_MAP_NEXT0 (test, PYBIND11_MAP_COMMA next, 0)
#endif
#define PYBIND11_MAP_LIST_NEXT(test, next) \
    PYBIND11_MAP_LIST_NEXT1 (PYBIND11_MAP_GET_END test, next)
#define PYBIND11_MAP_LIST0(f, t, x, peek, ...) \
    f(t, x) PYBIND11_MAP_LIST_NEXT (peek, PYBIND11_MAP_LIST1) (f, t, peek, __VA_ARGS__)
#define PYBIND11_MAP_LIST1(f, t, x, peek, ...) \
    f(t, x) PYBIND11_MAP_LIST_NEXT (peek, PYBIND11_MAP_LIST0) (f, t, peek, __VA_ARGS__)
// PYBIND11_MAP_LIST(f, t, a1, a2, ...) expands to f(t, a1), f(t, a2), ...
#define PYBIND11_MAP_LIST(f, t, ...) \
    PYBIND11_EVAL (PYBIND11_MAP_LIST1 (f, t, __VA_ARGS__, (), 0))

#define PYBIND11_NUMPY_DTYPE(Type, ...) \
    ::pybind11::detail::npy_format_descriptor<Type>::register_dtype \
        ({PYBIND11_MAP_LIST (PYBIND11_FIELD_DESCRIPTOR, Type, __VA_ARGS__)})

template  <class T>
using array_iterator = typename std::add_pointer<T>::type;

template <class T>
array_iterator<T> array_begin(const buffer_info& buffer) {
    return array_iterator<T>(reinterpret_cast<T*>(buffer.ptr));
}

template <class T>
array_iterator<T> array_end(const buffer_info& buffer) {
    return array_iterator<T>(reinterpret_cast<T*>(buffer.ptr) + buffer.size);
}

class common_iterator {
public:
    using container_type = std::vector<size_t>;
    using value_type = container_type::value_type;
    using size_type = container_type::size_type;

    common_iterator() : p_ptr(0), m_strides() {}

    common_iterator(void* ptr, const container_type& strides, const std::vector<size_t>& shape)
        : p_ptr(reinterpret_cast<char*>(ptr)), m_strides(strides.size()) {
        m_strides.back() = static_cast<value_type>(strides.back());
        for (size_type i = m_strides.size() - 1; i != 0; --i) {
            size_type j = i - 1;
            value_type s = static_cast<value_type>(shape[i]);
            m_strides[j] = strides[j] + m_strides[i] - strides[i] * s;
        }
    }

    void increment(size_type dim) {
        p_ptr += m_strides[dim];
    }

    void* data() const {
        return p_ptr;
    }

private:
    char* p_ptr;
    container_type m_strides;
};

template <size_t N> class multi_array_iterator {
public:
    using container_type = std::vector<size_t>;

    multi_array_iterator(const std::array<buffer_info, N> &buffers,
                         const std::vector<size_t> &shape)
        : m_shape(shape.size()), m_index(shape.size(), 0),
          m_common_iterator() {

        // Manual copy to avoid conversion warning if using std::copy
        for (size_t i = 0; i < shape.size(); ++i)
            m_shape[i] = static_cast<container_type::value_type>(shape[i]);

        container_type strides(shape.size());
        for (size_t i = 0; i < N; ++i)
            init_common_iterator(buffers[i], shape, m_common_iterator[i], strides);
    }

    multi_array_iterator& operator++() {
        for (size_t j = m_index.size(); j != 0; --j) {
            size_t i = j - 1;
            if (++m_index[i] != m_shape[i]) {
                increment_common_iterator(i);
                break;
            } else {
                m_index[i] = 0;
            }
        }
        return *this;
    }

    template <size_t K, class T> const T& data() const {
        return *reinterpret_cast<T*>(m_common_iterator[K].data());
    }

private:

    using common_iter = common_iterator;

    void init_common_iterator(const buffer_info &buffer,
                              const std::vector<size_t> &shape,
                              common_iter &iterator, container_type &strides) {
        auto buffer_shape_iter = buffer.shape.rbegin();
        auto buffer_strides_iter = buffer.strides.rbegin();
        auto shape_iter = shape.rbegin();
        auto strides_iter = strides.rbegin();

        while (buffer_shape_iter != buffer.shape.rend()) {
            if (*shape_iter == *buffer_shape_iter)
                *strides_iter = static_cast<size_t>(*buffer_strides_iter);
            else
                *strides_iter = 0;

            ++buffer_shape_iter;
            ++buffer_strides_iter;
            ++shape_iter;
            ++strides_iter;
        }

        std::fill(strides_iter, strides.rend(), 0);
        iterator = common_iter(buffer.ptr, strides, shape);
    }

    void increment_common_iterator(size_t dim) {
        for (auto &iter : m_common_iterator)
            iter.increment(dim);
    }

    container_type m_shape;
    container_type m_index;
    std::array<common_iter, N> m_common_iterator;
};

template <size_t N>
bool broadcast(const std::array<buffer_info, N>& buffers, size_t& ndim, std::vector<size_t>& shape) {
    ndim = std::accumulate(buffers.begin(), buffers.end(), size_t(0), [](size_t res, const buffer_info& buf) {
        return std::max(res, buf.ndim);
    });

    shape = std::vector<size_t>(ndim, 1);
    bool trivial_broadcast = true;
    for (size_t i = 0; i < N; ++i) {
        auto res_iter = shape.rbegin();
        bool i_trivial_broadcast = (buffers[i].size == 1) || (buffers[i].ndim == ndim);
        for (auto shape_iter = buffers[i].shape.rbegin();
             shape_iter != buffers[i].shape.rend(); ++shape_iter, ++res_iter) {

            if (*res_iter == 1)
                *res_iter = *shape_iter;
            else if ((*shape_iter != 1) && (*res_iter != *shape_iter))
                pybind11_fail("pybind11::vectorize: incompatible size/dimension of inputs!");

            i_trivial_broadcast = i_trivial_broadcast && (*res_iter == *shape_iter);
        }
        trivial_broadcast = trivial_broadcast && i_trivial_broadcast;
    }
    return trivial_broadcast;
}

template <typename Func, typename Return, typename... Args>
struct vectorize_helper {
    typename std::remove_reference<Func>::type f;

    template <typename T>
    vectorize_helper(T&&f) : f(std::forward<T>(f)) { }

    object operator()(array_t<Args>... args) {
        return run(args..., typename make_index_sequence<sizeof...(Args)>::type());
    }

    template <size_t ... Index> object run(array_t<Args>&... args, index_sequence<Index...> index) {
        /* Request buffers from all parameters */
        const size_t N = sizeof...(Args);

        std::array<buffer_info, N> buffers {{ args.request()... }};

        /* Determine dimensions parameters of output array */
        size_t ndim = 0;
        std::vector<size_t> shape(0);
        bool trivial_broadcast = broadcast(buffers, ndim, shape);

        size_t size = 1;
        std::vector<size_t> strides(ndim);
        if (ndim > 0) {
            strides[ndim-1] = sizeof(Return);
            for (size_t i = ndim - 1; i > 0; --i) {
                strides[i - 1] = strides[i] * shape[i];
                size *= shape[i];
            }
            size *= shape[0];
        }

        if (size == 1)
            return cast(f(*((Args *) buffers[Index].ptr)...));

        array_t<Return> result(shape, strides);
        auto buf = result.request();
        auto output = (Return *) buf.ptr;

        if (trivial_broadcast) {
            /* Call the function */
            for (size_t i = 0; i < size; ++i) {
                output[i] = f((buffers[Index].size == 1
                               ? *((Args *) buffers[Index].ptr)
                               : ((Args *) buffers[Index].ptr)[i])...);
            }
        } else {
            apply_broadcast<N, Index...>(buffers, buf, index);
        }

        return result;
    }

    template <size_t N, size_t... Index>
    void apply_broadcast(const std::array<buffer_info, N> &buffers,
                         buffer_info &output, index_sequence<Index...>) {
        using input_iterator = multi_array_iterator<N>;
        using output_iterator = array_iterator<Return>;

        input_iterator input_iter(buffers, output.shape);
        output_iterator output_end = array_end<Return>(output);

        for (output_iterator iter = array_begin<Return>(output);
             iter != output_end; ++iter, ++input_iter) {
            *iter = f((input_iter.template data<Index, Args>())...);
        }
    }
};

template <typename T, int Flags> struct handle_type_name<array_t<T, Flags>> {
    static PYBIND11_DESCR name() { return _("numpy.ndarray[") + type_caster<T>::name() + _("]"); }
};

NAMESPACE_END(detail)

template <typename Func, typename Return, typename... Args>
detail::vectorize_helper<Func, Return, Args...> vectorize(const Func &f, Return (*) (Args ...)) {
    return detail::vectorize_helper<Func, Return, Args...>(f);
}

template <typename Return, typename... Args>
detail::vectorize_helper<Return (*) (Args ...), Return, Args...> vectorize(Return (*f) (Args ...)) {
    return vectorize<Return (*) (Args ...), Return, Args...>(f, f);
}

template <typename func> auto vectorize(func &&f) -> decltype(
        vectorize(std::forward<func>(f), (typename detail::remove_class<decltype(&std::remove_reference<func>::type::operator())>::type *) nullptr)) {
    return vectorize(std::forward<func>(f), (typename detail::remove_class<decltype(
                   &std::remove_reference<func>::type::operator())>::type *) nullptr);
}

NAMESPACE_END(pybind11)

#if defined(_MSC_VER)
#pragma warning(pop)
#endif
