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
    //elpa_set(handle, "na", na, &error); // size of the na x na matrix
    int error;
    elpa_set(handle, varname, value, &error);
    return Py_BuildValue("i", error);
    //if(error != ELPA_OK) {
    //    PyErr_SetObject(PyExc_RuntimeError,
    //                    "Error in elpa allocate");
    //    return NULL;
    // }
    //Py_RETURN
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
    if (err != ELPA_OK) {
        PyErr_SetString(PyExc_RuntimeError,
                        "Error in elpa allocate");
        return NULL;
    }
    Py_RETURN_NONE;
}


PyObject* pyelpa_setup(PyObject *self, PyObject *args)
{
    //printf("pyelpa_setup\n");
    PyObject *handle_obj;
    if (!PyArg_ParseTuple(args, "O", &handle_obj))
        return NULL;

    elpa_t handle = unpack_handle(handle_obj);
    int err = elpa_setup(handle);
    if (err != ELPA_OK) {
        PyErr_SetString(PyExc_RuntimeError,
                        "Error in elpa allocate");
        return NULL;
    }
    Py_RETURN_NONE;
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

    int error;
    elpa_eigenvectors(handle, a, ev, q, &error);
    return Py_BuildValue("i", error);
}

PyObject* pyelpa_general_diagonalize(PyObject *self, PyObject *args)
{
    //printf("pyelpa_general_diagonalize\n");
    PyObject *handle_obj;
    PyArrayObject *A_obj, *S_obj, *C_obj, *eps_obj;

    if (!PyArg_ParseTuple(args,
                          "OOOOO",
                          &handle_obj,
                          &A_obj,
                          &S_obj,
                          &C_obj,
                          &eps_obj))
        return NULL;

    elpa_t handle = unpack_handle(handle_obj);
    //int error;
    //elpa_set(handle, "solver", ELPA_SOLVER_1STAGE, &error);
    //assert(error == ELPA_OK);


    int is_already_decomposed = 0;
    int error;

    double *ev = (double*)PyArray_DATA(eps_obj);

    if(PyArray_DESCR(A_obj)->type_num == NPY_DOUBLE) {
        double *a = (double*)PyArray_DATA(A_obj);
        double *b = (double*)PyArray_DATA(S_obj);
        double *q = (double*)PyArray_DATA(C_obj);
        elpa_generalized_eigenvectors(handle, a, b, ev, q,
                                      is_already_decomposed, &error);

    } else {
        double complex *a = (double complex *)PyArray_DATA(A_obj);
        double complex *b = (double complex *)PyArray_DATA(S_obj);
        double complex *q = (double complex *)PyArray_DATA(C_obj);
        elpa_generalized_eigenvectors(handle, a, b, ev, q,
                                      is_already_decomposed, &error);
    }

    return Py_BuildValue("i", error);
}


PyObject* pyelpa_diagonalize1(PyObject *self, PyObject *args)
{
    // printf("pyelpa_diagonalize\n");
    PyObject *gpaw_comm_obj;
    PyArrayObject *A, *C, *eps;
    int na, context, nev, blocksize, npcol, nprow, mycol, myrow, locM, locN;
    if (!PyArg_ParseTuple(args,
                          "OiOOOii(iiii)(ii)i",
                          &gpaw_comm_obj,
                          &context,
                          &A, &C, &eps,
                          &na, &nev,
                          &npcol, &nprow, &mycol, &myrow,
                          &locM, &locN, &blocksize))
        return NULL;

    MPIObject* gpaw_comm = (MPIObject *)gpaw_comm_obj;
    MPI_Comm comm = gpaw_comm->comm;

    if (elpa_init(20171201) != ELPA_OK) {
        // What API versions do we support?
        PyErr_SetString(PyExc_RuntimeError, "Error: ELPA API version not supported");
        return NULL;
    }

    int error = 0;

    int commsize;
    MPI_Comm_size(comm, &commsize);

    elpa_t handle = elpa_allocate(&error);

    elpa_set(handle, "na", na, &error); // size of the na x na matrix
    elpa_set(handle, "nev", nev, &error); // number of eigenvectors that should be computed ( 1<= nev <= na)
    printf("loc1 %d loc2 %d :: myrow %d mycol %d :: blk %d\n", locM, locN, myrow, mycol, blocksize);
    elpa_set(handle, "local_ncols", locM, &error);
    elpa_set(handle, "local_nrows", locN, &error); // number of local columns of the distributed matrix on this MPI task
    elpa_set(handle, "nblk", blocksize, &error);
    elpa_set(handle, "mpi_comm_parent", MPI_Comm_c2f(comm), &error); // the global MPI communicator
    elpa_set(handle, "process_row", myrow, &error); // row coordinate of MPI process
    elpa_set(handle, "process_col", mycol, &error);  // column coordinate of MPI process
    elpa_set(handle, "blacs_context", context, &error);


    error = elpa_setup(handle);

    elpa_set(handle, "solver", ELPA_SOLVER_1STAGE, &error);

    double *a = (double*)PyArray_DATA(A);
    double *ev = (double*)PyArray_DATA(eps);
    double *q = (double*)PyArray_DATA(C);

    elpa_eigenvectors(handle, a, ev, q, &error);
    elpa_deallocate(handle);
    return Py_BuildValue("i", error);
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
    Py_RETURN_NONE;
}

#endif
