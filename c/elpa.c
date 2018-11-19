#if defined(GPAW_WITH_SL) && defined(PARALLEL) && defined(GPAW_WITH_ELPA)

#include "extensions.h"
#include <elpa/elpa.h>
#include <mpi.h>
#include "mympi.h"

elpa_t* unpack_handle(PyObject* handle_obj)
{
    elpa_t* elpa = (elpa_t *)PyArray_DATA((PyArrayObject *)handle_obj);
    return elpa;
}

PyObject* pyelpa_set(PyObject *self, PyObject *args)
{
    printf("pyelpa_set\n");
    PyObject *handle_obj;
    char* varname;
    int value;
    if (!PyArg_ParseTuple(args, "Osi",
                          &handle_obj,
                          &varname,
                          &value)) {
        return NULL;
    }
    elpa_t* handle = unpack_handle(handle_obj);
    //elpa_set(handle, "na", na, &error); // size of the na x na matrix
    int error;
    elpa_set(*handle, varname, value, &error);
    return Py_BuildValue("i", error);
    //if(error != ELPA_OK) {
    //    PyErr_SetObject(PyExc_RuntimeError,
    //                    "Error in elpa allocate");
    //    return NULL;
    // }
    //Py_RETURN
}

PyObject* pyelpa_init(PyObject *self, PyObject *args)
{
    printf("pyelpa_init\n");
    PyObject *handle_obj;
    if (!PyArg_ParseTuple(args, "O", &handle_obj))
        return NULL;

    elpa_t *handle = unpack_handle(handle_obj);
    int err = 0;
    handle[0] = elpa_allocate(&err);
    if (err != ELPA_OK) {
        PyErr_SetString(PyExc_RuntimeError,
                        "Error in elpa allocate");
        return NULL;
    }
    Py_RETURN_NONE;
}


PyObject* pyelpa_diagonalize(PyObject *self, PyObject *args)
{
    printf("pyelpa_diagonalize\n");
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
    printf("pyelpa_deallocate\n");
    elpa_t* handle = unpack_handle(handle_obj);
    elpa_deallocate(*handle);
    Py_RETURN_NONE;
}

PyObject* pyelpa_general_diagonalize(PyObject *self, PyObject *args)
{
    printf("pyelpa_general_diagonalize\n");
    return NULL;
}


#endif
