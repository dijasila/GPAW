/*  Copyright (C) 2003-2007  CAMP
 *  Copyright (C) 2007-2009  CAMd
 *  Copyright (C) 2007-2010  CSC - IT Center for Science Ltd.
 *  Please see the accompanying LICENSE file for further information. */

#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL GPAW_ARRAY_API
#include <numpy/arrayobject.h>
#ifdef PARALLEL
#include <mpi.h>
#endif
#ifndef GPAW_WITHOUT_LIBXC
#include <xc.h>
#endif

#ifdef GPAW_HPM
PyObject* ibm_hpm_start(PyObject *self, PyObject *args);
PyObject* ibm_hpm_stop(PyObject *self, PyObject *args);
PyObject* ibm_mpi_start(PyObject *self);
PyObject* ibm_mpi_stop(PyObject *self);
#endif

#ifdef CRAYPAT
#include <pat_api.h>
PyObject* craypat_region_begin(PyObject *self, PyObject *args);
PyObject* craypat_region_end(PyObject *self, PyObject *args);
#endif

PyObject* symmetrize(PyObject *self, PyObject *args);
PyObject* symmetrize_ft(PyObject *self, PyObject *args);
PyObject* symmetrize_wavefunction(PyObject *self, PyObject *args);
PyObject* symmetrize_return_index(PyObject *self, PyObject *args);
PyObject* symmetrize_with_index(PyObject *self, PyObject *args);
PyObject* map_k_points(PyObject *self, PyObject *args);
PyObject* tetrahedron_weight(PyObject *self, PyObject *args);
#ifndef GPAW_WITHOUT_BLAS
PyObject* mmm(PyObject *self, PyObject *args);
PyObject* gemm(PyObject *self, PyObject *args);
PyObject* rk(PyObject *self, PyObject *args);
PyObject* r2k(PyObject *self, PyObject *args);
#endif
PyObject* NewOperatorObject(PyObject *self, PyObject *args);
PyObject* NewWOperatorObject(PyObject *self, PyObject *args);
PyObject* NewSplineObject(PyObject *self, PyObject *args);
PyObject* NewTransformerObject(PyObject *self, PyObject *args);
PyObject* pc_potential(PyObject *self, PyObject *args);
PyObject* add_to_density(PyObject *self, PyObject *args);
PyObject* utilities_gaussian_wave(PyObject *self, PyObject *args);
PyObject* pack(PyObject *self, PyObject *args);
PyObject* unpack(PyObject *self, PyObject *args);
PyObject* unpack_complex(PyObject *self, PyObject *args);
PyObject* hartree(PyObject *self, PyObject *args);
PyObject* localize(PyObject *self, PyObject *args);
PyObject* NewXCFunctionalObject(PyObject *self, PyObject *args);
#ifndef GPAW_WITHOUT_LIBXC
PyObject* NewlxcXCFunctionalObject(PyObject *self, PyObject *args);
PyObject* lxcXCFuncNum(PyObject *self, PyObject *args);
#endif
PyObject* exterior_electron_density_region(PyObject *self, PyObject *args);
PyObject* plane_wave_grid(PyObject *self, PyObject *args);
PyObject* tci_overlap(PyObject *self, PyObject *args);
PyObject *pwlfc_expand(PyObject *self, PyObject *args);
PyObject *pw_insert(PyObject *self, PyObject *args);
PyObject *pw_precond(PyObject *self, PyObject *args);
PyObject *fd_precond(PyObject *self, PyObject *args);
PyObject* vdw(PyObject *self, PyObject *args);
PyObject* vdw2(PyObject *self, PyObject *args);
PyObject* spherical_harmonics(PyObject *self, PyObject *args);
PyObject* spline_to_grid(PyObject *self, PyObject *args);
PyObject* NewLFCObject(PyObject *self, PyObject *args);
PyObject* globally_broadcast_bytes(PyObject *self, PyObject *args);
#if defined(GPAW_WITH_SL) && defined(PARALLEL)
PyObject* new_blacs_context(PyObject *self, PyObject *args);
PyObject* get_blacs_gridinfo(PyObject* self, PyObject *args);
PyObject* get_blacs_local_shape(PyObject* self, PyObject *args);
PyObject* blacs_destroy(PyObject *self, PyObject *args);
PyObject* scalapack_set(PyObject *self, PyObject *args);
PyObject* scalapack_redist(PyObject *self, PyObject *args);
PyObject* scalapack_diagonalize_dc(PyObject *self, PyObject *args);
PyObject* scalapack_diagonalize_ex(PyObject *self, PyObject *args);
#ifdef GPAW_MR3
PyObject* scalapack_diagonalize_mr3(PyObject *self, PyObject *args);
#endif
PyObject* scalapack_general_diagonalize_dc(PyObject *self, PyObject *args);
PyObject* scalapack_general_diagonalize_ex(PyObject *self, PyObject *args);
#ifdef GPAW_MR3
PyObject* scalapack_general_diagonalize_mr3(PyObject *self, PyObject *args);
#endif
PyObject* scalapack_inverse_cholesky(PyObject *self, PyObject *args);
PyObject* scalapack_inverse(PyObject *self, PyObject *args);
PyObject* scalapack_solve(PyObject *self, PyObject *args);
PyObject* pblas_tran(PyObject *self, PyObject *args);
PyObject* pblas_gemm(PyObject *self, PyObject *args);
PyObject* pblas_hemm_symm(PyObject *self, PyObject *args);
PyObject* pblas_gemv(PyObject *self, PyObject *args);
PyObject* pblas_r2k(PyObject *self, PyObject *args);
PyObject* pblas_rk(PyObject *self, PyObject *args);
#if defined(GPAW_WITH_ELPA)
#include <elpa/elpa.h>
PyObject* pyelpa_init(PyObject *self, PyObject *args);
PyObject* pyelpa_uninit(PyObject *self, PyObject *args);
PyObject* pyelpa_version(PyObject *self, PyObject *args);
PyObject* pyelpa_allocate(PyObject *self, PyObject *args);
PyObject* pyelpa_set(PyObject *self, PyObject *args);
PyObject* pyelpa_set_comm(PyObject *self, PyObject *args);
PyObject* pyelpa_setup(PyObject *self, PyObject *args);
PyObject* pyelpa_diagonalize(PyObject *self, PyObject *args);
PyObject* pyelpa_general_diagonalize(PyObject *self, PyObject *args);
PyObject* pyelpa_hermitian_multiply(PyObject *self, PyObject *args);
PyObject* pyelpa_constants(PyObject *self, PyObject *args);
PyObject* pyelpa_deallocate(PyObject *self, PyObject *args);
#endif // GPAW_WITH_ELPA
#endif // GPAW_WITH_SL and PARALLEL

#ifdef GPAW_WITH_FFTW
PyObject * FFTWPlan(PyObject *self, PyObject *args);
PyObject * FFTWExecute(PyObject *self, PyObject *args);
PyObject * FFTWDestroy(PyObject *self, PyObject *args);
#endif

// Threading
PyObject* get_num_threads(PyObject *self, PyObject *args);

#ifdef GPAW_PAPI
PyObject* papi_mem_info(PyObject *self, PyObject *args);
#endif

#ifdef GPAW_WITH_LIBVDWXC
PyObject* libvdwxc_create(PyObject *self, PyObject *args);
PyObject* libvdwxc_has(PyObject* self, PyObject *args);
PyObject* libvdwxc_init_serial(PyObject *self, PyObject *args);
PyObject* libvdwxc_calculate(PyObject *self, PyObject *args);
PyObject* libvdwxc_tostring(PyObject *self, PyObject *args);
PyObject* libvdwxc_free(PyObject* self, PyObject* args);
PyObject* libvdwxc_init_mpi(PyObject* self, PyObject* args);
PyObject* libvdwxc_init_pfft(PyObject* self, PyObject* args);
#endif // GPAW_WITH_LIBVDWXC

#ifdef GPAW_GITHASH
// For converting contents of a macro to a string, see
// https://en.wikipedia.org/wiki/C_preprocessor#Token_stringification
#define STR(s) #s
#define XSTR(s) STR(s)
PyObject* githash(PyObject* self, PyObject* args)
{
    return Py_BuildValue("s", XSTR(GPAW_GITHASH));
}
#undef XSTR
#undef STR
#endif // GPAW_GITHASH

// Holonomic constraints
PyObject* adjust_positions(PyObject *self, PyObject *args);
PyObject* adjust_momenta(PyObject *self, PyObject *args);
// TIP3P forces
PyObject* calculate_forces_H2O(PyObject *self, PyObject *args);


static PyMethodDef functions[] = {
    {"symmetrize", symmetrize, METH_VARARGS, 0},
    {"symmetrize_ft", symmetrize_ft, METH_VARARGS, 0},
    {"symmetrize_wavefunction", symmetrize_wavefunction, METH_VARARGS, 0},
    {"symmetrize_return_index", symmetrize_return_index, METH_VARARGS, 0},
    {"symmetrize_with_index", symmetrize_with_index, METH_VARARGS, 0},
    {"map_k_points", map_k_points, METH_VARARGS, 0},
    {"tetrahedron_weight", tetrahedron_weight, METH_VARARGS, 0},
#ifndef GPAW_WITHOUT_BLAS
    {"mmm", mmm, METH_VARARGS, 0},
    {"gemm", gemm, METH_VARARGS, 0},
    {"rk",  rk,  METH_VARARGS, 0},
    {"r2k", r2k, METH_VARARGS, 0},
#endif
    {"Operator", NewOperatorObject, METH_VARARGS, 0},
    {"WOperator", NewWOperatorObject, METH_VARARGS, 0},
    {"Spline", NewSplineObject, METH_VARARGS, 0},
    {"Transformer", NewTransformerObject, METH_VARARGS, 0},
    {"add_to_density", add_to_density, METH_VARARGS, 0},
    {"utilities_gaussian_wave", utilities_gaussian_wave, METH_VARARGS, 0},
    {"eed_region", exterior_electron_density_region, METH_VARARGS, 0},
    {"plane_wave_grid", plane_wave_grid, METH_VARARGS, 0},
    {"pwlfc_expand", pwlfc_expand, METH_VARARGS, 0},
    {"pw_insert", pw_insert, METH_VARARGS, 0},
    {"pw_precond", pw_precond, METH_VARARGS, 0},
    {"fd_precond", fd_precond, METH_VARARGS, 0},
    {"pack", pack, METH_VARARGS, 0},
    {"unpack", unpack, METH_VARARGS, 0},
    {"unpack_complex", unpack_complex,           METH_VARARGS, 0},
    {"hartree", hartree, METH_VARARGS, 0},
    {"localize", localize, METH_VARARGS, 0},
    {"XCFunctional", NewXCFunctionalObject, METH_VARARGS, 0},
#ifndef GPAW_WITHOUT_LIBXC
    {"lxcXCFunctional", NewlxcXCFunctionalObject, METH_VARARGS, 0},
    {"lxcXCFuncNum", lxcXCFuncNum, METH_VARARGS, 0},
#endif
    {"tci_overlap", tci_overlap, METH_VARARGS, 0},
    {"vdw", vdw, METH_VARARGS, 0},
    {"vdw2", vdw2, METH_VARARGS, 0},
    {"spherical_harmonics", spherical_harmonics, METH_VARARGS, 0},
    {"pc_potential", pc_potential, METH_VARARGS, 0},
    {"spline_to_grid", spline_to_grid, METH_VARARGS, 0},
    {"LFC", NewLFCObject, METH_VARARGS, 0},
    {"globally_broadcast_bytes", globally_broadcast_bytes, METH_VARARGS, 0},
    {"get_num_threads", get_num_threads, METH_VARARGS, 0},
#if defined(GPAW_WITH_SL) && defined(PARALLEL)
    {"new_blacs_context", new_blacs_context, METH_VARARGS, NULL},
    {"get_blacs_gridinfo", get_blacs_gridinfo, METH_VARARGS, NULL},
    {"get_blacs_local_shape", get_blacs_local_shape, METH_VARARGS, NULL},
    {"blacs_destroy", blacs_destroy, METH_VARARGS, 0},
    {"scalapack_set", scalapack_set, METH_VARARGS, 0},
    {"scalapack_redist", scalapack_redist, METH_VARARGS, 0},
    {"scalapack_diagonalize_dc", scalapack_diagonalize_dc, METH_VARARGS, 0},
    {"scalapack_diagonalize_ex", scalapack_diagonalize_ex, METH_VARARGS, 0},
#ifdef GPAW_MR3
    {"scalapack_diagonalize_mr3", scalapack_diagonalize_mr3, METH_VARARGS, 0},
#endif // GPAW_MR3
    {"scalapack_general_diagonalize_dc",
     scalapack_general_diagonalize_dc, METH_VARARGS, 0},
    {"scalapack_general_diagonalize_ex",
     scalapack_general_diagonalize_ex, METH_VARARGS, 0},
#ifdef GPAW_MR3
    {"scalapack_general_diagonalize_mr3",
     scalapack_general_diagonalize_mr3, METH_VARARGS, 0},
#endif // GPAW_MR3
    {"scalapack_inverse_cholesky", scalapack_inverse_cholesky,
     METH_VARARGS, 0},
    {"scalapack_inverse", scalapack_inverse, METH_VARARGS, 0},
    {"scalapack_solve", scalapack_solve, METH_VARARGS, 0},
    {"pblas_tran", pblas_tran, METH_VARARGS, 0},
    {"pblas_gemm", pblas_gemm, METH_VARARGS, 0},
    {"pblas_hemm_symm", pblas_hemm_symm, METH_VARARGS, 0},
    {"pblas_gemv", pblas_gemv, METH_VARARGS, 0},
    {"pblas_r2k", pblas_r2k, METH_VARARGS, 0},
    {"pblas_rk", pblas_rk, METH_VARARGS, 0},
#if defined(GPAW_WITH_ELPA)
    {"pyelpa_init", pyelpa_init, METH_VARARGS, 0},
    {"pyelpa_uninit", pyelpa_uninit, METH_VARARGS, 0},
    {"pyelpa_version", pyelpa_version, METH_VARARGS, 0},
    {"pyelpa_allocate", pyelpa_allocate, METH_VARARGS, 0},
    {"pyelpa_set", pyelpa_set, METH_VARARGS, 0},
    {"pyelpa_setup", pyelpa_setup, METH_VARARGS, 0},
    {"pyelpa_set_comm", pyelpa_set_comm, METH_VARARGS, 0},
    {"pyelpa_diagonalize", pyelpa_diagonalize, METH_VARARGS, 0},
    {"pyelpa_general_diagonalize", pyelpa_general_diagonalize, METH_VARARGS, 0},
    {"pyelpa_hermitian_multiply", pyelpa_hermitian_multiply, METH_VARARGS, 0},
    {"pyelpa_constants", pyelpa_constants, METH_VARARGS, 0},
    {"pyelpa_deallocate", pyelpa_deallocate, METH_VARARGS, 0},
#endif // GPAW_WITH_ELPA
#endif // GPAW_WITH_SL && PARALLEL
#ifdef GPAW_WITH_FFTW
    {"FFTWPlan", FFTWPlan, METH_VARARGS, 0},
    {"FFTWExecute", FFTWExecute, METH_VARARGS, 0},
    {"FFTWDestroy", FFTWDestroy, METH_VARARGS, 0},
#endif
#ifdef GPAW_HPM
    {"hpm_start", ibm_hpm_start, METH_VARARGS, 0},
    {"hpm_stop", ibm_hpm_stop, METH_VARARGS, 0},
    {"mpi_start", (PyCFunction) ibm_mpi_start, METH_NOARGS, 0},
    {"mpi_stop", (PyCFunction) ibm_mpi_stop, METH_NOARGS, 0},
#endif // GPAW_HPM
#ifdef CRAYPAT
    {"craypat_region_begin", craypat_region_begin, METH_VARARGS, 0},
    {"craypat_region_end", craypat_region_end, METH_VARARGS, 0},
#endif // CRAYPAT
#ifdef GPAW_PAPI
    {"papi_mem_info", papi_mem_info, METH_VARARGS, 0},
#endif // GPAW_PAPI
#ifdef GPAW_WITH_LIBVDWXC
    {"libvdwxc_create", libvdwxc_create, METH_VARARGS, 0},
    {"libvdwxc_has", libvdwxc_has, METH_VARARGS, 0},
    {"libvdwxc_init_serial", libvdwxc_init_serial, METH_VARARGS, 0},
    {"libvdwxc_calculate", libvdwxc_calculate, METH_VARARGS, 0},
    {"libvdwxc_tostring", libvdwxc_tostring, METH_VARARGS, 0},
    {"libvdwxc_free", libvdwxc_free, METH_VARARGS, 0},
    {"libvdwxc_init_mpi", libvdwxc_init_mpi, METH_VARARGS, 0},
    {"libvdwxc_init_pfft", libvdwxc_init_pfft, METH_VARARGS, 0},
#endif // GPAW_WITH_LIBVDWXC
    {"adjust_positions", adjust_positions, METH_VARARGS, 0},
    {"adjust_momenta", adjust_momenta, METH_VARARGS, 0},
    {"calculate_forces_H2O", calculate_forces_H2O, METH_VARARGS, 0},
#ifdef GPAW_GITHASH
    {"githash", githash, METH_VARARGS, 0},
#endif // GPAW_GITHASH
    {0, 0, 0, 0}
};

#ifdef PARALLEL
extern PyTypeObject MPIType;
extern PyTypeObject GPAW_MPI_Request_type;
#endif

extern PyTypeObject LFCType;
extern PyTypeObject OperatorType;
extern PyTypeObject WOperatorType;
extern PyTypeObject SplineType;
extern PyTypeObject TransformerType;
extern PyTypeObject XCFunctionalType;
#ifndef GPAW_WITHOUT_LIBXC
extern PyTypeObject lxcXCFunctionalType;
#endif

PyObject* globally_broadcast_bytes(PyObject *self, PyObject *args)
{
    PyObject *pybytes;
    if(!PyArg_ParseTuple(args, "O", &pybytes)){
        return NULL;
    }

#ifdef PARALLEL
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank;
    MPI_Comm_rank(comm, &rank);

    long size;
    if(rank == 0) {
        size = PyBytes_Size(pybytes);  // Py_ssize_t --> long
    }
    MPI_Bcast(&size, 1, MPI_LONG, 0, comm);

    char *dst = (char *)malloc(size);
    if(rank == 0) {
        char *src = PyBytes_AsString(pybytes);  // Read-only
        memcpy(dst, src, size);
    }
    MPI_Bcast(dst, size, MPI_BYTE, 0, comm);

    PyObject *value = PyBytes_FromStringAndSize(dst, size);
    free(dst);
    return value;
#else
    return pybytes;
#endif
}


static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_gpaw",
    "C-extension for GPAW",
    -1,
    functions,
    NULL,
    NULL,
    NULL,
    NULL
};

static PyObject* moduleinit(void)
{
#ifdef PARALLEL
    if (PyType_Ready(&MPIType) < 0)
        return NULL;
    if (PyType_Ready(&GPAW_MPI_Request_type) < 0)
        return NULL;
#endif

    if (PyType_Ready(&LFCType) < 0)
        return NULL;
    if (PyType_Ready(&OperatorType) < 0)
        return NULL;
    if (PyType_Ready(&WOperatorType) < 0)
        return NULL;
    if (PyType_Ready(&SplineType) < 0)
        return NULL;
    if (PyType_Ready(&TransformerType) < 0)
        return NULL;
    if (PyType_Ready(&XCFunctionalType) < 0)
        return NULL;
#ifndef GPAW_WITHOUT_LIBXC
    if (PyType_Ready(&lxcXCFunctionalType) < 0)
        return NULL;
#endif

    PyObject* m = PyModule_Create(&moduledef);

    if (m == NULL)
        return NULL;

#ifdef PARALLEL
    Py_INCREF(&MPIType);
    Py_INCREF(&GPAW_MPI_Request_type);
    PyModule_AddObject(m, "Communicator", (PyObject *)&MPIType);
#endif

#ifndef GPAW_WITHOUT_LIBXC
# if XC_MAJOR_VERSION >= 3
    PyObject_SetAttrString(m,
                           "libxc_version",
                           PyUnicode_FromString(xc_version_string()));
# endif
#endif
#ifdef _OPENMP
    PyObject_SetAttrString(m, "have_openmp", Py_True);
#else
    PyObject_SetAttrString(m, "have_openmp", Py_False);
#endif

    Py_INCREF(&LFCType);
    Py_INCREF(&OperatorType);
    Py_INCREF(&WOperatorType);
    Py_INCREF(&SplineType);
    Py_INCREF(&TransformerType);
    Py_INCREF(&XCFunctionalType);
#ifndef GPAW_WITHOUT_LIBXC
    Py_INCREF(&lxcXCFunctionalType);
#endif
#ifndef GPAW_INTERPRETER
    // gpaw-python needs to import arrays at the right time, so this is
    // done in gpaw_main().  In serial, we just do it here:
    import_array1(0);
#endif
    return m;
}

#ifndef GPAW_INTERPRETER


PyMODINIT_FUNC PyInit__gpaw(void)
{
    return moduleinit();
}

#else // ifndef GPAW_INTERPRETER

int
gpaw_main()
{
    int status = -1;

    PyObject *gpaw_mod = NULL, *pymain = NULL;

    gpaw_mod = PyImport_ImportModule("gpaw");
    if(gpaw_mod == NULL) {
        status = 3;  // Basic import failure
    } else {
        pymain = PyObject_GetAttrString(gpaw_mod, "main");
    }

    if(pymain == NULL) {
        status = 4;  // gpaw.main does not exist for some reason
        //PyErr_Print();
    } else {
        // Returns Py_None or NULL (error after calling user script)
        // We already imported the Python parts of numpy.  If we want, we can
        // later attempt to broadcast the numpy C API imports, too.
        // However I don't know how many files they are, and we need to
        // figure out how to broadcast extension modules (shared objects).
        import_array1(0);
        PyObject *pyreturn = PyObject_CallFunction(pymain, "");
        status = (pyreturn == NULL);
        Py_XDECREF(pyreturn);
    }

    Py_XDECREF(pymain);
    Py_XDECREF(gpaw_mod);
    return status;
}


int
main(int argc, char **argv)
{
#ifndef _OPENMP
    MPI_Init(&argc, &argv);
#else
    int granted;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &granted);
    if (granted != MPI_THREAD_MULTIPLE)
        exit(1);
#endif 

#define PyChar wchar_t
    wchar_t* wargv[argc];
    wchar_t* wargv2[argc];
    for (int i = 0; i < argc; i++) {
        int n = 1 + mbstowcs(NULL, argv[i], 0);
        wargv[i] = (wchar_t*)malloc(n * sizeof(wchar_t));
        wargv2[i] = wargv[i];
        mbstowcs(wargv[i], argv[i], n);
    }

    Py_SetProgramName(wargv[0]);
    PyImport_AppendInittab("_gpaw", &moduleinit);
    Py_Initialize();
    PySys_SetArgvEx(argc, wargv, 0);

#ifdef GPAW_WITH_ELPA
    // Globally initialize Elpa library if present:
    if (elpa_init(20171201) != ELPA_OK) {
        // What API versions do we support?
        PyErr_SetString(PyExc_RuntimeError, "Elpa >= 20171201 required");
        PyErr_Print();
        return 1;
    }
#endif

    int status = gpaw_main();

    if(status != 0) {
        PyErr_Print();
    }

#ifdef GPAW_WITH_ELPA

#ifdef ELPA_API_VERSION
    // Newer Elpas define their version but older ones don't.
    int elpa_err;
    elpa_uninit(&elpa_err);
#else
    elpa_uninit();  // 2018.05.001: no errcode
#endif

#endif

    Py_Finalize();
    MPI_Finalize();

    for (int i = 0; i < argc; i++)
        free(wargv2[i]);

    return status;
}
#endif // GPAW_INTERPRETER
