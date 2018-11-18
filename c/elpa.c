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

PyObject* elpa_initxxx(PyObject *self, PyObject *args)
{
    PyObject *handle_obj;
    if (!PyArg_ParseTuple(args, "O", &handle_obj))
        return NULL;

    // ......... init

    elpa_t *handle = unpack_handle(handle_obj);
    int err = 0;
    handle[0] = elpa_allocate(&err);
    if (err) {
        PyErr_SetString(PyExc_RuntimeError,
                        "Error in elpa allocate");
        return NULL;
    }
    Py_RETURN_NONE;
}

PyObject* elpa_general_diagonalize(PyObject *self, PyObject *args)
{

    PyObject *gpaw_comm_obj;
    PyArrayObject *A, *S, *C, *eps;
    int na, context, nev, blocksize, npcol, nprow, mycol, myrow, locM, locN;
    if (!PyArg_ParseTuple(args,
                          "OiOOOOii(iiii)(ii)i",
                          &gpaw_comm_obj,
                          &context,
                          &A, &S, &C, &eps,
                          &na, &nev,
                          &npcol, &nprow, &mycol, &myrow,
                          &locM, &locN, &blocksize))
        return NULL;

    MPIObject* gpaw_comm = (MPIObject *)gpaw_comm_obj;
    MPI_Comm comm = gpaw_comm->comm;

    double *grumble = PyArray_DATA(A);
    //printf("%f\n", grumble[0]);


    //int mpi_fcomm = MPI_Comm_c2f(MPI_COMM_WORLD);
    //int my_blacs_ctxt;
    //int sc_desc[9];
    //int info;
    //set_up_blacsgrid_f(mpi_fcomm, nprow, npcol, 'C', &my_blacs_ctxt,
    //                   &myrow, &mycol);
    //set_up_blacs_descriptor_f(na, blocksize, myrow, mycol, nprow, npcol, &locM, &locN, sc_desc, my_blacs_ctxt, &info);

    if (elpa_init(20171201) != ELPA_OK) {
        // What API versions do we support?
        PyErr_SetString(PyExc_RuntimeError, "Error: ELPA API version not supported");
        return NULL;
    }

    int error = 0;

    int commsize;
    MPI_Comm_size(comm, &commsize);

    elpa_t handle = elpa_allocate(&error);
    //assert_elpa_ok(error);

    //int na_rows = na;
    //int na_cols = na;
    //int nblk = 2;
    //int my_prow = 0; // MPI Cartesian grid
    //int my_pcol = 0;

    /* Set parameters the matrix and it's MPI distribution */
    elpa_set(handle, "na", na, &error); // size of the na x na matrix
    elpa_set(handle, "nev", nev, &error); // number of eigenvectors that should be computed ( 1<= nev <= na)
    //elpa_set(handle, "local_nrows", nprow, &error);
    printf("loc1 %d loc2 %d :: myrow %d mycol %d :: blk %d\n", locM, locN, myrow, mycol, blocksize);
    elpa_set(handle, "local_ncols", locM, &error);
    elpa_set(handle, "local_nrows", locN, &error); // number of local columns of the distributed matrix on this MPI task
    elpa_set(handle, "nblk", blocksize, &error);
    elpa_set(handle, "mpi_comm_parent", MPI_Comm_c2f(comm), &error); // the global MPI communicator
    elpa_set(handle, "process_row", myrow, &error); // row coordinate of MPI process
    elpa_set(handle, "process_col", mycol, &error);  // column coordinate of MPI process
    elpa_set(handle, "blacs_context", context, &error);

    /* Setup */
    error = elpa_setup(handle);

    /* if desired, set any number of tunable run-time options */
    /* look at the list of possible options as detailed later in
       USERS_GUIDE.md */

    elpa_set(handle, "solver", ELPA_SOLVER_2STAGE, &error);

    /* use method solve to solve the eigenvalue problem */
    /* other possible methods are desribed in USERS_GUIDE.md */

    double *a = (double*)PyArray_DATA(A);
    double *b = (double*)PyArray_DATA(S);
    double *ev = (double*)PyArray_DATA(eps);
    double *q = (double*)PyArray_DATA(C);
    //printf("grrr %f %f %f %f\n",
    //       a[0], a[1], a[2], a[3]);
    int already_decomposed = 0;
    //elpa_generalized_eigenvectors(handle, a, b, ev, q,
    //                              already_decomposed, &error);
    elpa_eigenvectors(handle, a, ev, q, &error);
    //printf("Err %d\n", error);
    //if (error) {
    //    PyErr_SetString(PyExc_RuntimeError,
    //                    "Error getting eigenvectors");
    //    return NULL;
    //}

    /* cleanup */
    elpa_deallocate(handle);
    //elpa_uninit();
    return Py_BuildValue("i", error);
    //Py_RETURN_NONE;
}

#endif
