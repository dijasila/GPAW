#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL GPAW_ARRAY_API
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>
#include <stdlib.h>
#include <pthread.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include "../extensions.h"
#define __TRANSFORMERS_C
#include "../transformers.h"
#undef __TRANSFORMERS_C
#include "gpaw-cuda.h"

extern int gpaw_cuda_debug;

static double *transformer_buf_gpu = NULL;
static int transformer_buf_size = 0;
static int transformer_init_count = 0;

static int debug_size_in = 0;
static int debug_size_out = 0;
static int debug_size_buf = 0;
static int debug_size_buf_out = 0;
static double *debug_sendbuf;
static double *debug_recvbuf;
static double *debug_buf_cpu;
static double *debug_buf_out;
static double *debug_out_cpu;
static double *debug_out_gpu;
static double *debug_in_cpu;

/*
 * Increment reference count to register a new tranformer object.
 */
void transformer_init_cuda(TransformerObject *self)
{
    transformer_init_count++;
}

/*
 * Ensure buffer is allocated and is big enough. Reallocate only if
 * size has increased.
 */
void transformer_init_buffers(TransformerObject *self, int blocks)
{
    const boundary_conditions* bc = self->bc;
    const int* size2 = bc->size2;
    int ng2 = (bc->ndouble * size2[0] * size2[1] * size2[2]) * blocks;

    if (ng2 > transformer_buf_size) {
        cudaFree(transformer_buf_gpu);
        cudaGetLastError();
        GPAW_CUDAMALLOC(&transformer_buf_gpu, double, ng2);
        transformer_buf_size = ng2;
    }
}

/*
 * Reset reference count and unset buffer.
 */
void transformer_init_buffers_cuda()
{
    transformer_buf_gpu = NULL;
    transformer_buf_size = 0;
    transformer_init_count = 0;
}

/*
 * Deallocate buffer or decrease reference count.
 *
 * arguments:
 *   (int) force -- if true, force deallocation
 */
void transformer_dealloc_cuda(int force)
{
    if (force)
        transformer_init_count = 1;

    if (transformer_init_count == 1) {
        cudaFree(transformer_buf_gpu);
        cudaGetLastError();
        transformer_init_buffers_cuda();
        return;
    }
    if (transformer_init_count > 0)
        transformer_init_count--;
}

/*
 * Allocate debug buffers and precalculate sizes.
 */
void debug_transformer_allocate(TransformerObject* self, int nin, int blocks)
{
    boundary_conditions* bc = self->bc;
    const int *size1 = bc->size1;
    const int *size2 = bc->size2;
    int ng = bc->ndouble * size1[0] * size1[1] * size1[2];
    int ng2 = bc->ndouble * size2[0] * size2[1] * size2[2];
    int out_ng = bc->ndouble * self->size_out[0] * self->size_out[1]
               * self->size_out[2];

    debug_size_in = ng * nin;
    debug_size_out = out_ng * nin;
    debug_size_buf = ng2 * blocks;
    debug_size_buf_out = MAX(out_ng, ng2) * blocks;

    debug_sendbuf = GPAW_MALLOC(double, bc->maxsend * blocks * bc->ndouble);
    debug_recvbuf = GPAW_MALLOC(double, bc->maxrecv * blocks * bc->ndouble);
    debug_buf_cpu = GPAW_MALLOC(double, debug_size_buf);
    debug_buf_out = GPAW_MALLOC(double, debug_size_buf_out);
    debug_out_cpu = GPAW_MALLOC(double, debug_size_out);
    debug_out_gpu = GPAW_MALLOC(double, debug_size_out);
    debug_in_cpu = GPAW_MALLOC(double, debug_size_in);
}

/*
 * Deallocate debug buffers and set sizes to zero.
 */
void debug_transformer_deallocate()
{
    free(debug_sendbuf);
    free(debug_recvbuf);
    free(debug_buf_cpu);
    free(debug_buf_out);
    free(debug_out_cpu);
    free(debug_out_gpu);
    free(debug_in_cpu);
    debug_size_in = 0;
    debug_size_out = 0;
    debug_size_buf = 0;
    debug_size_buf_out = 0;
}

/*
 * Copy initial GPU arrays to debug buffers on the CPU.
 */
void debug_transformer_memcpy_pre(const double *in, double *out)
{
    GPAW_CUDAMEMCPY(debug_in_cpu, in, double, debug_size_in,
                    cudaMemcpyDeviceToHost);
    GPAW_CUDAMEMCPY(debug_out_cpu, out, double, debug_size_out,
                    cudaMemcpyDeviceToHost);
}

/*
 * Copy final GPU arrays to debug buffers on the CPU.
 */
void debug_transformer_memcpy_post(double *out)
{
    GPAW_CUDAMEMCPY(debug_out_gpu, out, double, debug_size_out,
                    cudaMemcpyDeviceToHost);
}

/*
 * Run the interpolate and restrict algorithm (see transapply_worker()
 * in ../transformers.c) on the CPU and compare to results from the GPU.
 */
void debug_transformer_apply(TransformerObject* self,
                             int nin, int blocks, bool real,
                             const double_complex *ph)
{
    MPI_Request recvreq[2];
    MPI_Request sendreq[2];
    int i;
    double err;

    boundary_conditions* bc = self->bc;
    const int *size1 = bc->size1;
    const int *size2 = bc->size2;
    int ng = bc->ndouble * size1[0] * size1[1] * size1[2];
    int ng2 = bc->ndouble * size2[0] * size2[1] * size2[2];
    int out_ng = bc->ndouble * self->size_out[0] * self->size_out[1]
               * self->size_out[2];
    int rank = 0;
    if (bc->comm != MPI_COMM_NULL)
        MPI_Comm_rank(bc->comm, &rank);

    int last;
    for (int n=0; n < nin; n += blocks) {
        const double *in = debug_in_cpu + n * ng;
        double *out = debug_out_cpu + n * out_ng;
        int myblocks = MIN(blocks, nin - n);
        last = myblocks;
        for (int i=0; i < 3; i++) {
            bc_unpack1(bc, in, debug_buf_cpu, i, recvreq, sendreq,
                       debug_recvbuf, debug_sendbuf, ph + 2 * i, 0, myblocks);
            bc_unpack2(bc, debug_buf_cpu, i, recvreq, sendreq,
                       debug_recvbuf, myblocks);
        }
        for (int m=0; m < myblocks; m++) {
            if (real) {
                if (self->interpolate)
                    bmgs_interpolate(self->k, self->skip,
                                     debug_buf_cpu + m * ng2,
                                     bc->size2, debug_out_cpu + m * out_ng,
                                     debug_buf_out + m * MAX(out_ng, ng2));
                else
                    bmgs_restrict(self->k, debug_buf_cpu + m * ng2,
                                  bc->size2, debug_out_cpu + m * out_ng,
                                  debug_buf_out + m * MAX(out_ng, ng2));
            } else {
                if (self->interpolate)
                    bmgs_interpolatez(
                            self->k, self->skip,
                            (double_complex*) (debug_buf_cpu + m * ng2),
                            bc->size2,
                            (double_complex*) (debug_out_cpu + m * out_ng),
                            (double_complex*) (debug_buf_out
                                               + m * MAX(out_ng, ng2)));
                else
                    bmgs_restrictz(
                            self->k,
                            (double_complex*) (debug_buf_cpu + m * ng2),
                            bc->size2,
                            (double_complex*) (debug_out_cpu + m * out_ng),
                            (double_complex*) (debug_buf_out
                                               + m * MAX(out_ng, ng2)));
            }
        }
    }

    double out_err = 0;
    for (i=0; i < debug_size_out; i++) {
        out_err = MAX(out_err, fabs(debug_out_cpu[i] - debug_out_gpu[i]));
    }
    if (out_err > GPAW_CUDA_ABS_TOL) {
        fprintf(stderr,
                "[%d] Debug CUDA transformer apply (out): error %g\n",
                rank, out_err);
    }
}

/*
 * Run the interpolate and restrict algorithm (see transapply_worker()
 * in ../transformers.c) on the GPU.
 */
static void _transformer_apply_cuda_gpu(TransformerObject* self,
                                        const double *in, double *out,
                                        int nin, int blocks, bool real,
                                        const double_complex *ph)
{
    boundary_conditions* bc = self->bc;
    const int* size1 = bc->size1;
    const int* size2 = self->bc->size2;
    int ng = bc->ndouble * size1[0] * size1[1] * size1[2];
    int ng2 = bc->ndouble * size2[0] * size2[1] * size2[2];
    int out_ng = bc->ndouble * self->size_out[0] * self->size_out[1]
               * self->size_out[2];

    int mpi_size = 1;
    if ((bc->maxsend || bc->maxrecv) && bc->comm != MPI_COMM_NULL)
        MPI_Comm_size(bc->comm, &mpi_size);

    MPI_Request recvreq[3][2];
    MPI_Request sendreq[3][2];

    transformer_init_buffers(self, blocks);

    double* buf = transformer_buf_gpu;

    for (int n = 0; n < nin; n += blocks) {
        const double* in2 = in + n * ng;
        double* out2 = out + n * out_ng;
        int myblocks = MIN(blocks, nin - n);

        bc_unpack_paste_cuda_gpu(bc, in2, buf, recvreq, 0, myblocks);
        for (int i=0; i < 3; i++) {
            bc_unpack_cuda_gpu(bc, in2, buf, i, recvreq, sendreq[i],
                               ph + 2 * i, 0, myblocks);
        }
        if (self->interpolate) {
            if (real) {
                bmgs_interpolate_cuda_gpu(self->k, self->skip, buf,
                                          bc->size2, out2, self->size_out,
                                          myblocks);
            } else {
                bmgs_interpolate_cuda_gpuz(self->k, self->skip,
                                           (cuDoubleComplex*) (buf),
                                           bc->size2,
                                           (cuDoubleComplex*) (out2),
                                           self->size_out, myblocks);
            }
        } else {
            if (real) {
                bmgs_restrict_cuda_gpu(self->k, buf, bc->size2,
                                       out2, self->size_out, myblocks);
            } else {
                bmgs_restrict_cuda_gpuz(self->k,
                                        (cuDoubleComplex*) (buf),
                                        bc->size2,
                                        (cuDoubleComplex*) (out2),
                                        self->size_out, myblocks);
            }
        }
    }
}

/*
 * Python interface for the GPU version of the interpolate and restrict
 * algorithm (similar to Transformer_apply() for CPUs).
 *
 * arguments:
 *   input_gpu  -- pointer to device memory (GPUArray.gpudata)
 *   output_gpu -- pointer to device memory (GPUArray.gpudata)
 *   shape      -- shape of the array (tuple)
 *   type       -- datatype of array elements
 *   phases     -- phase (complex) (ignored if type is NPY_DOUBLE)
 */
PyObject* Transformer_apply_cuda_gpu(TransformerObject *self, PyObject *args)
{
    PyArrayObject* phases = 0;
    CUdeviceptr input_gpu;
    CUdeviceptr output_gpu;
    PyObject *shape;
    PyArray_Descr *type;

    if (!PyArg_ParseTuple(args, "nnOO|O", &input_gpu, &output_gpu, &shape,
                          &type, &phases))
        return NULL;

    int nin = 1;
    if (PyTuple_Size(shape) == 4)
        nin = (int) PyLong_AsLong(PyTuple_GetItem(shape, 0));

    const double* in = (double*) input_gpu;
    double* out = (double*) output_gpu;

    bool real = (type->type_num == NPY_DOUBLE);
    const double_complex* ph = (real ? 0 : COMPLEXP(phases));

    boundary_conditions* bc = self->bc;
    int mpi_size = 1;
    if ((bc->maxsend || bc->maxrecv) && bc->comm != MPI_COMM_NULL)
        MPI_Comm_size(bc->comm, &mpi_size);

    int blocks = MAX(1, MIN(nin, MIN((GPAW_CUDA_BLOCKS_MIN) * mpi_size,
                                     (GPAW_CUDA_BLOCKS_MAX) / bc->ndouble)));

    if (gpaw_cuda_debug) {
        debug_transformer_allocate(self, nin, blocks);
        debug_transformer_memcpy_pre(in, out);
    }

    _transformer_apply_cuda_gpu(self, in, out, nin, blocks, real, ph);

    if (gpaw_cuda_debug) {
        cudaDeviceSynchronize();
        debug_transformer_memcpy_post(out);
        debug_transformer_apply(self, nin, blocks, real, ph);
        debug_transformer_deallocate();
    }

    if (PyErr_Occurred())
        return NULL;
    else
        Py_RETURN_NONE;
}
