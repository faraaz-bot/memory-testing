//
// Python extension module for hipFFT.
//

#include <exception>
#include <stdexcept>
#include <algorithm>

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL hipfft_ARRAY_API
#include <numpy/arrayobject.h>

#include <hipfft.h>

#define BEGIN_EXTERN_C extern "C" {
#define END_EXTERN_C }

#define HIPFFT_CHECK(ret)                                          \
    if((ret) != HIPFFT_SUCCESS)                                    \
    {                                                              \
        PyErr_SetObject(PyExc_RuntimeError, PyLong_FromLong(ret)); \
        return NULL;                                               \
    }

#define HIP_CHECK(ret)                                             \
    if((ret) != hipSuccess)                                        \
    {                                                              \
        PyErr_SetObject(PyExc_RuntimeError, PyLong_FromLong(ret)); \
        return NULL;                                               \
    }

BEGIN_EXTERN_C

typedef struct
{
    hipfftType fft_type;
    int        npy_type;
} transform_types_t;

static transform_types_t transform_types(PyArrayObject* x, bool real, int direction)
{
    if(PyArray_TYPE(x) == NPY_FLOAT32)
    {
        return {HIPFFT_R2C, NPY_COMPLEX64};
    }
    if(PyArray_TYPE(x) == NPY_FLOAT64)
    {
        return {HIPFFT_D2Z, NPY_COMPLEX128};
    }
    if(PyArray_TYPE(x) == NPY_COMPLEX64)
    {
        if(real && direction == HIPFFT_BACKWARD)
            return {HIPFFT_C2R, NPY_FLOAT32};
        return {HIPFFT_C2C, NPY_COMPLEX64};
    }
    if(PyArray_TYPE(x) == NPY_COMPLEX128)
    {
        if(real && direction == HIPFFT_BACKWARD)
            return {HIPFFT_Z2D, NPY_FLOAT64};
        return {HIPFFT_Z2Z, NPY_COMPLEX128};
    }
    throw std::runtime_error("FFT type cannot be deduced.");
}

static PyObject* hipfft_transform(PyObject* X, bool real, int direction, bool batched, bool time)
{
    hipEvent_t start, stop;
    float      elapsed = 1.0;
    if(time)
    {
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));
    }

    PyArrayObject* x  = (PyArrayObject*)X;
    npy_intp       nd = 0, nb = 0, nx = 0, ny = 0, nz = 0;

    if(batched)
    {
        nd = PyArray_NDIM(x) - 1;
        nb = PyArray_DIM(x, 0);
        nx = PyArray_DIM(x, 1);
        ny = (nd > 1) ? PyArray_DIM(x, 2) : 1;
        nz = (nd > 2) ? PyArray_DIM(x, 3) : 1;
    }
    else
    {
        nd = PyArray_NDIM(x);
        nb = 1;
        nx = PyArray_DIM(x, 0);
        ny = (nd > 1) ? PyArray_DIM(x, 1) : 1;
        nz = (nd > 2) ? PyArray_DIM(x, 2) : 1;
    }

    if(real && direction == HIPFFT_BACKWARD)
    {
        if(nd == 1)
            nx = 2 * (nx - 1);
        if(nd == 2)
            ny = 2 * (ny - 1);
        if(nd == 3)
            nz = 2 * (nz - 1);
    }

    // cout << "nd " << nd << endl;
    // cout << "nb " << nb << endl;
    // cout << "nx " << nx << endl;
    // cout << "ny " << ny << endl;
    // cout << "nz " << nz << endl;

    hipfftHandle      plan;
    transform_types_t type;
    try
    {
        type = transform_types(x, real, direction);
    }
    catch(std::exception& e)
    {
        HIPFFT_CHECK(HIPFFT_INVALID_TYPE);
    }

    int n[3] = {int(nx), int(ny), int(nz)};
    HIPFFT_CHECK(
        hipfftPlanMany(&plan, int(nd), n, nullptr, 1, 0, nullptr, 1, 0, type.fft_type, int(nb)));

    PyObject* Z;
    if(nb > 1)
    {
        npy_intp dims[4] = {nb, nx, ny, nz};
        if(type.fft_type == HIPFFT_R2C || type.fft_type == HIPFFT_D2Z)
            dims[nd] = dims[nd] / 2 + 1;
        Z = PyArray_SimpleNew(nd + 1, dims, type.npy_type);
    }
    else
    {
        npy_intp dims[3] = {nx, ny, nz};
        if(type.fft_type == HIPFFT_R2C || type.fft_type == HIPFFT_D2Z)
            dims[nd - 1] = dims[nd - 1] / 2 + 1;
        Z = PyArray_SimpleNew(nd, dims, type.npy_type);
    }
    PyArrayObject* z = (PyArrayObject*)Z;

    size_t total_bytes_in  = (size_t)PyArray_NBYTES(x);
    size_t total_bytes_out = (size_t)PyArray_NBYTES(z);
    void  *d_in, *d_out;
    HIP_CHECK(hipMalloc(&d_in, total_bytes_in));
    HIP_CHECK(hipMalloc(&d_out, total_bytes_out));
    HIP_CHECK(hipMemcpy(d_in, PyArray_DATA(x), total_bytes_in, hipMemcpyHostToDevice));

    if(time)
    {
        HIP_CHECK(hipEventRecord(start, 0));
    }

    switch(type.fft_type)
    {
    case HIPFFT_C2C:
        HIPFFT_CHECK(
            hipfftExecC2C(plan, (hipfftComplex*)d_in, (hipfftComplex*)d_out, direction));
        break;
    case HIPFFT_R2C:
        HIPFFT_CHECK(hipfftExecR2C(plan, (hipfftReal*)d_in, (hipfftComplex*)d_out));
        break;
    case HIPFFT_C2R:
        HIPFFT_CHECK(hipfftExecC2R(plan, (hipfftComplex*)d_in, (hipfftReal*)d_out));
        break;
    case HIPFFT_D2Z:
        HIPFFT_CHECK(
            hipfftExecD2Z(plan, (hipfftDoubleReal*)d_in, (hipfftDoubleComplex*)d_out));
        break;
    case HIPFFT_Z2D:
        HIPFFT_CHECK(
            hipfftExecZ2D(plan, (hipfftDoubleComplex*)d_in, (hipfftDoubleReal*)d_out));
        break;
    case HIPFFT_Z2Z:
        HIPFFT_CHECK(hipfftExecZ2Z(
            plan, (hipfftDoubleComplex*)d_in, (hipfftDoubleComplex*)d_out, direction));
        break;
    default:
        HIPFFT_CHECK(HIPFFT_INVALID_TYPE);
        break;
    }

    if(time)
    {
        HIP_CHECK(hipEventRecord(stop, 0));
        HIP_CHECK(hipEventSynchronize(stop));
        HIP_CHECK(hipEventElapsedTime(&elapsed, start, stop));
        HIP_CHECK(hipEventDestroy(stop));
        HIP_CHECK(hipEventDestroy(start));
    }

    HIP_CHECK(hipMemcpy(PyArray_DATA(z), d_out, total_bytes_out, hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_out));
    HIPFFT_CHECK(hipfftDestroy(plan));

    if(time)
    {
        PyObject *R = Py_BuildValue("Of", Z, elapsed);
        Py_XDECREF(Z);
        return R;
    }

    return Z;
}

static PyObject* hipfft_forward(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* X;
    int       real = 0, batched = 0, time = 0;

    static const char* kwlist[] = {"x", "real", "batched", "time", NULL};
    if(!PyArg_ParseTupleAndKeywords(
           args, kwargs, "O|ppp", (char**)kwlist, &X, &real, &batched, &time))
        return NULL;

    if(!PyArray_CheckExact(X))
        return NULL; // better messaging...

    return hipfft_transform(X, bool(real), HIPFFT_FORWARD, bool(batched), bool(time));
}

static PyObject* hipfft_backward(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* X;
    int       real = 0, batched = 0, time = 0;

    static const char* kwlist[] = {"x", "real", "batched", "time", NULL};
    if(!PyArg_ParseTupleAndKeywords(
           args, kwargs, "O|ppp", (char**)kwlist, &X, &real, &batched, &time))
        return NULL;

    if(!PyArray_CheckExact(X))
        return NULL; // better messaging...

    return hipfft_transform(X, bool(real), HIPFFT_BACKWARD, bool(batched), bool(time));
}

// clang-format off
static PyMethodDef hipfft_methods[] = {
  {"forward", (PyCFunction)(void (*)(void))hipfft_forward, METH_VARARGS | METH_KEYWORDS, "Forward FFT."},
  {"backward", (PyCFunction)(void (*)(void))hipfft_backward, METH_VARARGS | METH_KEYWORDS, "Inverse/backward FFT."},
  {NULL, NULL, 0, NULL}
};
// clang-format on

static struct PyModuleDef hipfft_module
    = {PyModuleDef_HEAD_INIT, "hipfft", NULL, -1, hipfft_methods};

PyMODINIT_FUNC PyInit_hipfft(void)
{
    PyObject* m = PyModule_Create(&hipfft_module);
    if(m == NULL)
        return NULL;

    PyModule_AddIntConstant(m, "SUCCESS", HIPFFT_SUCCESS);
    PyModule_AddIntConstant(m, "INVALID_PLAN", HIPFFT_INVALID_PLAN);
    PyModule_AddIntConstant(m, "ALLOC_FAILED", HIPFFT_ALLOC_FAILED);
    PyModule_AddIntConstant(m, "INVALID_TYPE", HIPFFT_INVALID_TYPE);
    PyModule_AddIntConstant(m, "INVALID_VALUE", HIPFFT_INVALID_VALUE);
    PyModule_AddIntConstant(m, "INTERNAL_ERROR", HIPFFT_INTERNAL_ERROR);
    PyModule_AddIntConstant(m, "EXEC_FAILED", HIPFFT_EXEC_FAILED);
    PyModule_AddIntConstant(m, "SETUP_FAILED", HIPFFT_SETUP_FAILED);
    PyModule_AddIntConstant(m, "INVALID_SIZE", HIPFFT_INVALID_SIZE);
    PyModule_AddIntConstant(m, "UNALIGNED_DATA", HIPFFT_UNALIGNED_DATA);
    PyModule_AddIntConstant(m, "INCOMPLETE_PARAMETER_LIST", HIPFFT_INCOMPLETE_PARAMETER_LIST);
    PyModule_AddIntConstant(m, "INVALID_DEVICE", HIPFFT_INVALID_DEVICE);
    PyModule_AddIntConstant(m, "PARSE_ERROR", HIPFFT_PARSE_ERROR);
    PyModule_AddIntConstant(m, "NO_WORKSPACE", HIPFFT_NO_WORKSPACE);
    PyModule_AddIntConstant(m, "NOT_IMPLEMENTED", HIPFFT_NOT_IMPLEMENTED);
    PyModule_AddIntConstant(m, "NOT_SUPPORTED", HIPFFT_NOT_SUPPORTED);

    PyModule_AddIntConstant(m, "R2C", HIPFFT_R2C);
    PyModule_AddIntConstant(m, "C2R", HIPFFT_C2R);
    PyModule_AddIntConstant(m, "C2C", HIPFFT_C2C);
    PyModule_AddIntConstant(m, "D2Z", HIPFFT_D2Z);
    PyModule_AddIntConstant(m, "Z2D", HIPFFT_Z2D);
    PyModule_AddIntConstant(m, "Z2Z", HIPFFT_Z2Z);

    PyModule_AddIntConstant(m, "FORWARD", HIPFFT_FORWARD);
    PyModule_AddIntConstant(m, "BACKWARD", HIPFFT_BACKWARD);

    import_array();

    return m;
}

END_EXTERN_C
