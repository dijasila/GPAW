#include <Python.h>

PyObject* compiled_WITH_HDF5(PyObject *self, PyObject *args)
{
     if (!PyArg_ParseTuple(args, ""))
          return NULL;

     int hdf5 = 0;

#ifdef GPAW_WITH_HDF5
     hdf5 = 1;
#endif

     return Py_BuildValue("i", hdf5);
}
