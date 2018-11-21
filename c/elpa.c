#if defined(GPAW_WITH_SL) && defined(PARALLEL) && defined(GPAW_WITH_ELPA)

#include "extensions.h"
#include <elpa/elpa.h>
#include <mpi.h>
#include "mympi.h"

elpa_t* unpack_handleptr(PyObject* handle_obj)
{
    elpa_t* elpa = (elpa_t *)PyArray_DATA((PyArrayObject *)handle_obj);
    return elpa;
}

elpa_t unpack_handle(PyObject* handle_obj)
{
    elpa_t* elpa = unpack_handleptr(handle_obj);
    return *elpa;
}



PyObject* pyelpa_set(PyObject *self, PyObject *args)
{
    //printf("pyelpa_set\n");
    PyObject *handle_obj;
    char* varname;
    int value;
    if (!PyArg_ParseTuple(args, "Osi",
                          &handle_obj,
                          &varname,
                          &value)) {
        return NULL;
    }
    elpa_t handle = unpack_handle(handle_obj);
    int err;
    elpa_set(handle, varname, value, &err);
    return Py_BuildValue("i", err);
}

PyObject* pyelpa_allocate(PyObject *self, PyObject *args)
{
    //printf("pyelpa_init\n");
    PyObject *handle_obj;
    if (!PyArg_ParseTuple(args, "O", &handle_obj))
        return NULL;

    elpa_t *handle = unpack_handleptr(handle_obj);
    int err = 0;
    handle[0] = elpa_allocate(&err);
    return Py_BuildValue("i", err);
}


PyObject* pyelpa_setup(PyObject *self, PyObject *args)
{
    //printf("pyelpa_setup\n");
    PyObject *handle_obj;
    if (!PyArg_ParseTuple(args, "O", &handle_obj))
        return NULL;

    elpa_t handle = unpack_handle(handle_obj);
    int err = elpa_setup(handle);
    return Py_BuildValue("i", err);
}


PyObject* pyelpa_set_comm(PyObject *self, PyObject *args)
{
    PyObject *handle_obj;
    PyObject *gpaw_comm_obj;

    if(!PyArg_ParseTuple(args, "OO", &handle_obj,
                         &gpaw_comm_obj))
        return NULL;
    elpa_t handle = unpack_handle(handle_obj);
    MPIObject *gpaw_comm = (MPIObject *)gpaw_comm_obj;
    MPI_Comm comm = gpaw_comm->comm;
    int fcomm = MPI_Comm_c2f(comm);
    int err;
    elpa_set(handle, "mpi_comm_parent", fcomm, &err);
    return Py_BuildValue("i", err);
}

PyObject* pyelpa_constants(PyObject *self, PyObject *args)
{
    if(!PyArg_ParseTuple(args, ""))
        return NULL;
    return Py_BuildValue("iii",
                         ELPA_OK,
                         ELPA_SOLVER_1STAGE,
                         ELPA_SOLVER_2STAGE);
}


PyObject* pyelpa_diagonalize(PyObject *self, PyObject *args)
{
    //printf("pyelpa_diagonalize\n");
    PyObject *handle_obj;
    PyArrayObject *A_obj, *C_obj, *eps_obj;

    if (!PyArg_ParseTuple(args,
                          "OOOO",
                          &handle_obj,
                          &A_obj,
                          &C_obj,
                          &eps_obj))
        return NULL;

    elpa_t handle = unpack_handle(handle_obj);

    double *a = (double*)PyArray_DATA(A_obj);
    double *ev = (double*)PyArray_DATA(eps_obj);
    double *q = (double*)PyArray_DATA(C_obj);

    int err;
    elpa_eigenvectors(handle, a, ev, q, &err);
    return Py_BuildValue("i", err);
}

PyObject* pyelpa_general_diagonalize(PyObject *self, PyObject *args)
{
    //printf("pyelpa_general_diagonalize\n");
    PyObject *handle_obj;
    PyArrayObject *A_obj, *S_obj, *C_obj, *eps_obj;
    int is_already_decomposed;

    if (!PyArg_ParseTuple(args,
                          "OOOOOi",
                          &handle_obj,
                          &A_obj,
                          &S_obj,
                          &C_obj,
                          &eps_obj,
                          &is_already_decomposed))
        return NULL;

    elpa_t handle = unpack_handle(handle_obj);

    int err;
    double *ev = (double*)PyArray_DATA(eps_obj);

    if(PyArray_DESCR(A_obj)->type_num == NPY_DOUBLE) {
        double *a = (double*)PyArray_DATA(A_obj);
        double *b = (double*)PyArray_DATA(S_obj);
        double *q = (double*)PyArray_DATA(C_obj);
        elpa_generalized_eigenvectors(handle, a, b, ev, q,
                                      is_already_decomposed, &err);

    } else {
        double complex *a = (double complex *)PyArray_DATA(A_obj);
        double complex *b = (double complex *)PyArray_DATA(S_obj);
        double complex *q = (double complex *)PyArray_DATA(C_obj);
        elpa_generalized_eigenvectors(handle, a, b, ev, q,
                                      is_already_decomposed, &err);
    }

    return Py_BuildValue("i", err);
}

PyObject *pyelpa_deallocate(PyObject *self, PyObject *args)
{
    PyObject *handle_obj;
    if(!PyArg_ParseTuple(args, "O", &handle_obj)) {
        return NULL;
    }
    //printf("pyelpa_deallocate\n");
    elpa_t handle = unpack_handle(handle_obj);
    elpa_deallocate(handle);
    // This function provides no error checking
    Py_RETURN_NONE;
}

#endif
