#ifdef GPAW_WITH_LIBVDWXC
#include "../extensions.h"

#include <vdwxc.h>

#ifdef PARALLEL
#include <mpi.h>
#include "../mympi.h"
#endif


// Our heinous plan is to abuse a numpy array so that it will contain a pointer to the vdw_data.
// This is because PyCapsules are not there until Python 3.1/2.7.
// This function takes an array and returns the pointer it so outrageously contains.
vdw_data* unpack_vdw_pointer(PyObject* vdw_obj)
{
    vdw_data* vdw = (vdw_data *)PyArray_DATA((PyArrayObject *)vdw_obj);
    return vdw;
}

PyObject* libvdwxc_create(PyObject* self, PyObject* args, PyObject* kwargs)
{
    int vdw_code;
    int Nx, Ny, Nz;
    double C00, C10, C20, C01, C11, C21, C02, C12, C22;

    if(!PyArg_ParseTuple(args, "i(iii)(ddddddddd)|Oi",
                         &vdw_code, // functional identifier
                         &Nx, &Ny, &Nz, // number of grid points
                         &C00, &C10, &C20, // 3x3 cell
                         &C01, &C11,&C21,
                         &C02, &C12, &C22)) {
        return NULL;
    }

    vdw_data vdw = vdw_new(vdw_code);
    vdw_set_unit_cell(vdw, Nx, Ny, Nz, C00, C10, C20, C01, C11, C21, C02, C12, C22);

    // That was the real work.  Now we create the horrific array to contain the pointer.
    npy_intp pointersize = sizeof(void *);
    PyObject* vdw_obj = PyArray_SimpleNew(1, &pointersize, NPY_BYTE);
    Py_INCREF(vdw_obj); // To be DECREF'd after we call vdw_finalize
    vdw_data* vdw_ptr = unpack_vdw_pointer(vdw_obj);
    vdw_ptr[0] = vdw;
    return vdw_obj;
}

PyObject* libvdwxc_init_serial(PyObject* self, PyObject* args)
{
    PyObject* vdw_obj;
    if(!PyArg_ParseTuple(args, "O", &vdw_obj)) {
        return NULL;
    }
    vdw_data* vdw = unpack_vdw_pointer(vdw_obj);
    vdw_init_serial(*vdw);
    Py_RETURN_NONE;
}    

PyObject* libvdwxc_calculate(PyObject* self, PyObject* args)
{
    PyObject *vdw_obj;
    PyArrayObject *rho_obj, *sigma_obj, *dedn_obj, *dedsigma_obj;
    if(!PyArg_ParseTuple(args, "OOOOO",
                         &vdw_obj, &rho_obj, &sigma_obj,
                         &dedn_obj, &dedsigma_obj)) {
        return NULL;
    }
    vdw_data* vdw = unpack_vdw_pointer(vdw_obj);
    double* rho_g = (double*)PyArray_DATA(rho_obj);
    double* sigma_g = (double*)PyArray_DATA(sigma_obj);
    double* dedn_g = (double*)PyArray_DATA(dedn_obj);
    double* dedsigma_g = (double*)PyArray_DATA(dedsigma_obj);

    double energy = vdw_calculate(*vdw, rho_g, sigma_g, dedn_g, dedsigma_g);
    return Py_BuildValue("d", energy);
}

PyObject* libvdwxc_free(PyObject* self, PyObject* args)
{
    PyObject* vdw_obj;
    if(!PyArg_ParseTuple(args, "O", &vdw_obj)) {
        return NULL;
    }
    vdw_data* vdw = unpack_vdw_pointer(vdw_obj);
    vdw_finalize(vdw);
    Py_DECREF(vdw_obj); // We INCREF'd this back when we created it.  Now it may be GC'd.
    Py_RETURN_NONE;
}

#ifdef PARALLEL
MPI_Comm unpack_gpaw_comm(PyObject* gpaw_mpi_obj)
{
    MPIObject* gpaw_comm = (MPIObject *)gpaw_mpi_obj;
    return gpaw_comm->comm;
}

PyObject* libvdwxc_init_mpi(PyObject* self, PyObject* args)
{
    PyObject* vdw_obj;
    PyObject* gpaw_comm_obj;
    if(!PyArg_ParseTuple(args, "OO", &vdw_obj, &gpaw_comm_obj)) {
        return NULL;
    }
    
    if(!vdw_has_fftw_mpi()) {
        return NULL;
    }

    vdw_data* vdw = unpack_vdw_pointer(vdw_obj);
    MPI_Comm comm = unpack_gpaw_comm(gpaw_comm_obj);
    vdw_init_mpi(*vdw, comm);
    Py_RETURN_NONE;
}

PyObject* libvdwxc_init_pfft(PyObject* self, PyObject* args)
{
    PyObject* vdw_obj;
    PyObject* gpaw_comm_obj;
    int nproc1, nproc2;
    if(!PyArg_ParseTuple(args, "OOii", &vdw_obj, &gpaw_comm_obj, &nproc1, &nproc2)) {
        return NULL;
    }
    
    if(!vdw_has_pfft) {
        return NULL;
    }

    vdw_data* vdw = unpack_vdw_pointer(vdw_obj);
    MPI_Comm comm = unpack_gpaw_comm(gpaw_comm_obj);
    vdw_init_pfft(*vdw, comm, nproc1, nproc2);
    Py_RETURN_NONE;
}

#endif // PARALLEL

#endif // gpaw_with_libvdwxc
