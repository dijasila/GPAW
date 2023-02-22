// In the code, one utilizes calls equvalent to PyArray API,
// except instead of PyArray_BYTES one uses Array_BYTES.
// Then, if GPAW is built with GPAW_GPU_AWARE_MPI define, these macros are rewritten with wrappers.
#ifndef GPAW_GPU_AWARE_MPI
// Check that array is well-behaved and contains data that can be sent.
#define CHK_ARRAY(a) if ((a) == NULL || !PyArray_Check(a)                   \
			 || !PyArray_ISCARRAY(a) || !PyArray_ISNUMBER(a)) { \
    PyErr_SetString(PyExc_TypeError,                                        \
		    "Not a proper NumPy array for MPI communication.");     \
    return NULL; } else

// Check that array is well-behaved, read-only  and contains data that
// can be sent.
#define CHK_ARRAY_RO(a) if ((a) == NULL || !PyArray_Check(a)                \
			 || !PyArray_ISCARRAY_RO(a)                         \
			 || !PyArray_ISNUMBER(a)) {                         \
    PyErr_SetString(PyExc_TypeError,                                        \
		    "Not a proper NumPy array for MPI communication.");     \
    return NULL; } else

// Check that two arrays have the same type, and the size of the
// second is a given multiple of the size of the first
#define CHK_ARRAYS(a,b,n)                                               \
  if ((PyArray_TYPE(a) != PyArray_TYPE(b))                              \
      || (PyArray_SIZE(b) != PyArray_SIZE(a) * n)) {                    \
    PyErr_SetString(PyExc_ValueError,                                   \
		    "Incompatible array types or sizes.");              \
      return NULL; } else


#define Array_NDIM(a) PyArray_NDIM(a)
#define Array_DIM(a,d)  PyArray_DIM(a,d)
#define Array_ITEMSIZE(a) PyArray_ITEMSIZE(a)
#define Array_BYTES(a) PyArray_BYTES(a)
#define Array_DATA(a) PyArray_DATA(a)
#define Array_SIZE(a) PyArray_SIZE(a)
#define Array_TYPE(a) PyArray_TYPE(a)
#define Array_NBYTES(a) PyArray_NBYTES(a)
#define Array_ISCOMPLEX(a) PyArray_ISCOMPLEX(a)

#else // GPAW_GPU_AWARE_MPI

#define CHK_ARRAY(a) // TODO
#define CHK_ARRAY_RO(a) // TODO
#define CHK_ARRAYS(a,b,n) // TODO

static int Array_NDIM(PyObject* obj)
{
    if (PyArray_Check(obj))
    {
	return PyArray_NDIM((PyArrayObject*)obj);
    }

    // return len(obj.shape)
    PyObject* shape = PyObject_GetAttrString(obj, "shape");
    if (shape == NULL) return -1;
    Py_DECREF(shape);
    return PyTuple_Size(shape);
}

static int Array_DIM(PyObject* obj, int dim)
{
    if (PyArray_Check(obj))
    {
	return PyArray_DIM((PyArrayObject*)obj, dim);
    }
    PyObject* shape_str = Py_BuildValue("s", "shape");
    PyObject* shape = PyObject_GetAttr(obj, shape_str);
    Py_DECREF(shape_str);

    if (shape == NULL) return -1;
    PyObject* pydim = PyTuple_GetItem(shape, dim);
    Py_DECREF(shape);
    if (pydim == NULL) return -1;
    return (int) PyLong_AS_LONG(pydim);
}

static char* Array_BYTES(PyObject* obj)
{
    if (PyArray_Check(obj))
    {
	return PyArray_BYTES((PyArrayObject*)obj);
    }
    //ndarray.data.ptr
    PyObject* ndarray_data = PyObject_GetAttrString(obj, "data");
    if (ndarray_data == NULL) return NULL;
    PyObject* ptr_data = PyObject_GetAttrString(ndarray_data, "ptr");
    if (ptr_data == NULL) return NULL;
    char* ptr = (char*) PyLong_AS_LONG(ptr_data);
    Py_DECREF(ptr_data);
    Py_DECREF(ndarray_data);
    return ptr;
}

#define Array_DATA(a) ((void*) Array_BYTES(a))

static int Array_SIZE(PyObject* obj)
{
    PyObject* size_str = Py_BuildValue("s", "size");
    PyObject* size = PyObject_GetAttr(obj, size_str);
    int arraysize = (int) PyLong_AS_LONG(size);
    Py_DECREF(size);
    Py_DECREF(size_str);
    return arraysize;
}

static int Array_TYPE(PyObject* obj)
{
    if (PyArray_Check(obj))
    {
	return PyArray_TYPE((PyArrayObject*)obj);
    }
    PyObject* dtype_str = Py_BuildValue("s", "dtype");
    PyObject* dtype = PyObject_GetAttr(obj, dtype_str);
    Py_DECREF(dtype_str);

    if (dtype == NULL) return -1;

    PyObject* num_str = Py_BuildValue("s", "num");
    PyObject* num = PyObject_GetAttr(dtype, num_str);
    Py_DECREF(num_str);
    Py_DECREF(dtype);
    if (num == NULL) return -1;

    int value =  (int) PyLong_AS_LONG(num);
    Py_DECREF(num);
    return value;
}

static int Array_ITEMSIZE(PyObject* obj)
{
    if (PyArray_Check(obj))
    {
	return PyArray_ITEMSIZE((PyArrayObject*)obj);
    }
    PyObject* dtype = PyObject_GetAttrString(obj, "dtype");
    if (dtype == NULL) return -1;
    PyObject* itemsize_obj = PyObject_GetAttrString(dtype, "itemsize");
    if (itemsize_obj == NULL) return -1;
    int itemsize = (int) PyLong_AS_LONG(itemsize_obj);
    Py_DECREF(itemsize_obj);
    Py_DECREF(dtype);
    return itemsize;
}


static long Array_NBYTES(PyObject* obj)
{
    if (PyArray_Check(obj))
    {
	return PyArray_NBYTES((PyArrayObject*)obj);
    }
    PyObject* nbytes_str = Py_BuildValue("s", "nbytes");
    PyObject* nbytes = PyObject_GetAttr(obj, nbytes_str);
    long nbytesvalue = PyLong_AS_LONG(nbytes);
    Py_DECREF(nbytes_str);
    Py_DECREF(nbytes);
    return nbytesvalue;
}

static int Array_ISCOMPLEX(PyObject* obj)
{
    int result = PyTypeNum_ISCOMPLEX(Array_TYPE(obj));
    return result;
}

#endif

