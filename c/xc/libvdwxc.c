#ifdef GPAW_WITH_LIBVDWXC
#include "../extensions.h"

#include <vdw_core.h>

#ifdef PARALLEL
#include <vdw_mpi.h>
#include "../mympi.h"
#endif


PyObject* libvdwxc_initialize(PyObject* self, PyObject* args)
{
    printf("vdw init\n");
    PyObject* comm;
    int Nx, Ny, Nz;
    double C00, C10, C20, C01, C11, C21, C02, C12, C22;
    if(!PyArg_ParseTuple(args, "Oiiiddddddddd", &comm,
                         &Nx, &Ny, &Nz,
                         &C00, &C10, &C20, &C01, &C11, &C21, &C02, &C12, &C22)) {
        return NULL;
    }
    npy_intp objsize = vdw_df_data_get_size();
    PyArrayObject* vdwdata_obj = (PyArrayObject*)PyArray_SimpleNew(1, &objsize, NPY_BYTE);
    void* vdwdata = PyArray_DATA(vdwdata_obj);
    vdw_df_initialize(vdwdata, FUNC_VDWDF);

    if(comm != Py_None) {
        PyErr_SetString(PyExc_RuntimeError, "comm given in serial.");
        return NULL; // comm should be None for serial calculation.
    }
    vdw_df_set_unit_cell(vdwdata, Nx, Ny, Nz, C00, C10, C20, C01, C11, C21, C02, C12, C22);
    return (PyObject*)vdwdata_obj;
}

PyObject* libvdwxc_calculate(PyObject* self, PyObject* args)
{
    PyArrayObject *vdw_obj, *rho_obj, *sigma_obj, *dedn_obj, *dedsigma_obj;
    if(!PyArg_ParseTuple(args, "OOOOO",
                         &vdw_obj, &rho_obj, &sigma_obj,
                         &dedn_obj, &dedsigma_obj)) {
        return NULL;
    }
    struct vdw_df_data* vdwdata = (struct vdw_df_data*)PyArray_DATA(vdw_obj);
    double* rho_g = (double*)PyArray_DATA(rho_obj);
    double* sigma_g = (double*)PyArray_DATA(sigma_obj);
    double* dedn_g = (double*)PyArray_DATA(dedn_obj);
    double* dedsigma_g = (double*)PyArray_DATA(dedsigma_obj);

    double energy = vdw_df_calculate(vdwdata, rho_g, sigma_g, dedn_g, dedsigma_g);
    return Py_BuildValue("d", energy);
}

PyObject* libvdwxc_free(PyObject* self, PyObject* args)
{
    PyArrayObject* vdw_obj;
    if(!PyArg_ParseTuple(args, "O", &vdw_obj)) {
        return NULL;
    }
    struct vdw_df_data* vdwdata = (struct vdw_df_data*)PyArray_DATA(vdw_obj);
    vdw_df_finalize(vdwdata);
    Py_RETURN_NONE;
}

#ifdef PARALLEL
PyObject* libvdwxc_initialize_mpi(PyObject* self, PyObject* args)
{
    printf("vdw mpi init\n");
    fftw_mpi_init(); // can be called as many times as one wants.
    PyObject* comm;
    int Nx, Ny, Nz;
    double C00, C10, C20, C01, C11, C21, C02, C12, C22;
    if(!PyArg_ParseTuple(args, "Oiiiddddddddd", &comm,
                         &Nx, &Ny, &Nz,
                         &C00, &C10, &C20, &C01, &C11, &C21, &C02, &C12, &C22)) {
        return NULL;
    }
    npy_intp objsize = vdw_df_data_get_size();
    PyArrayObject* vdwdata_obj = (PyArrayObject*)PyArray_SimpleNew(1, &objsize, NPY_BYTE);
    void* vdwdata = PyArray_DATA(vdwdata_obj);
    vdw_df_initialize(vdwdata, FUNC_VDWDF);

    MPIObject* gpaw_comm = (MPIObject *)comm;
    vdw_df_set_communicator(vdwdata, gpaw_comm->comm);
    vdw_df_set_unit_cell(vdwdata, Nx, Ny, Nz, C00, C10, C20, C01, C11, C21, C02, C12, C22);
    return (PyObject*)vdwdata_obj;
}

PyObject* libvdwxc_calculate_mpi(PyObject* self, PyObject* args)
{
    PyArrayObject *vdw_obj, *rho_obj, *sigma_obj, *dedn_obj, *dedsigma_obj;
    if(!PyArg_ParseTuple(args, "OOOOO",
                         &vdw_obj, &rho_obj, &sigma_obj,
                         &dedn_obj, &dedsigma_obj)) {
        return NULL;
    }
    struct vdw_df_data* vdwdata = (struct vdw_df_data*)PyArray_DATA(vdw_obj);
    double* rho_g = (double*)PyArray_DATA(rho_obj);
    double* sigma_g = (double*)PyArray_DATA(sigma_obj);
    double* dedn_g = (double*)PyArray_DATA(dedn_obj);
    double* dedsigma_g = (double*)PyArray_DATA(dedsigma_obj);

    double energy;
    if(vdwdata->use_mpi) {
        energy = vdw_df_mpi_calculate(vdwdata, rho_g, sigma_g, dedn_g, dedsigma_g);
    } else {
        energy = vdw_df_calculate(vdwdata, rho_g, sigma_g, dedn_g, dedsigma_g);
    }
    return Py_BuildValue("d", energy);
}

#else
PyObject* libvdwxc_calculate_mpi(PyObject* self, PyObject* args){
    printf("what the FUCK\n");
    return NULL;
}
PyObject* libvdwxc_initialize_mpi(PyObject* self, PyObject* args){
    printf("what the FUCK\n");
    return NULL;
}

#endif // parallel

#endif // gpaw_with_libvdwxc
