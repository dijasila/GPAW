#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL GPAW_ARRAY_API
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>
#include <stdlib.h>
#include <pthread.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <../extensions.h>
#define __OPERATORS_C
#include <../operators.h>
#undef __OPERATORS_C
#include "gpaw-cuda.h"

extern int gpaw_cuda_debug;

#define OPERATOR_NSTREAMS (2)

static cudaStream_t operator_stream[OPERATOR_NSTREAMS];
static cudaEvent_t operator_event[2];
static int operator_streams = 0;

static double *operator_buf_gpu = NULL;
static int operator_buf_size = 0;
static int operator_buf_max = 0;
static int operator_init_count = 0;

static double *debug_sendbuf;
static double *debug_recvbuf;
static double *debug_buf_cpu;
static double *debug_buf_gpu;
static double *debug_out_cpu;
static double *debug_out_gpu;
static double *debug_in_cpu;

void operator_init_cuda(OperatorObject *self)
{
    const boundary_conditions* bc = self->bc;
    const int* size2 = bc->size2;
    int ng2 = bc->ndouble * size2[0] * size2[1] * size2[2];

    operator_buf_max = MAX(ng2, operator_buf_max);

    self->stencil_gpu = bmgs_stencil_to_gpu(&(self->stencil));
    operator_init_count++;
}

void operator_alloc_buffers(OperatorObject *self, int blocks)
{
    const boundary_conditions* bc = self->bc;
    const int* size2 = bc->size2;
    int ng2 = (bc->ndouble * size2[0] * size2[1] * size2[2]) * blocks;

    operator_buf_max = MAX(ng2, operator_buf_max);

    if (operator_buf_max > operator_buf_size) {
        cudaFree(operator_buf_gpu);
        cudaGetLastError();
        GPAW_CUDAMALLOC(&operator_buf_gpu, double, operator_buf_max);
        operator_buf_size = operator_buf_max;
    }
    if (!operator_streams) {
        for (int i=0; i < OPERATOR_NSTREAMS; i++) {
            cudaStreamCreate(&(operator_stream[i]));
        }
        for (int i=0; i < 2; i++) {
            cudaEventCreateWithFlags(
                    &operator_event[i],
                    cudaEventDefault|cudaEventDisableTiming);
        }
        operator_streams = OPERATOR_NSTREAMS;
    }
}

void operator_init_buffers_cuda()
{
    operator_buf_gpu = NULL;
    operator_buf_size = 0;
    operator_init_count = 0;
    operator_streams = 0;
}

void operator_dealloc_cuda(int force)
{
    if (force) {
        operator_init_count = 1;
    }
    if (operator_init_count == 1) {
        cudaError_t rval;
        rval = cudaFree(operator_buf_gpu);
        if (rval == cudaSuccess && operator_streams) {
            for (int i=0; i < OPERATOR_NSTREAMS; i++) {
                gpaw_cudaSafeCall(cudaStreamSynchronize(operator_stream[i]));
                gpaw_cudaSafeCall(cudaStreamDestroy(operator_stream[i]));
            }
            for (int i=0; i < 2; i++) {
                gpaw_cudaSafeCall(cudaEventDestroy(operator_event[i]));
            }
        }
        operator_init_buffers_cuda();
        return;
    }
    if (operator_init_count > 0) {
        operator_init_count--;
    }
}

void debug_operator_allocate(boundary_conditions *bc, int ng, int ng2,
                             int blocks)
{
    debug_sendbuf = GPAW_MALLOC(double, bc->maxsend * blocks);
    debug_recvbuf = GPAW_MALLOC(double, bc->maxrecv * blocks);
    debug_buf_cpu = GPAW_MALLOC(double, ng2 * blocks);
    debug_buf_gpu = GPAW_MALLOC(double, ng2 * blocks);
    debug_out_cpu = GPAW_MALLOC(double, ng * blocks);
    debug_out_gpu = GPAW_MALLOC(double, ng * blocks);
    debug_in_cpu = GPAW_MALLOC(double, ng * blocks);
}

void debug_operator_deallocate()
{
    free(debug_sendbuf);
    free(debug_recvbuf);
    free(debug_buf_cpu);
    free(debug_buf_gpu);
    free(debug_out_cpu);
    free(debug_out_gpu);
    free(debug_in_cpu);
}

void debug_operator_memcpy_pre(double *out, const double *in, int ng)
{
    GPAW_CUDAMEMCPY(debug_in_cpu, in, double, ng, cudaMemcpyDeviceToHost);
    GPAW_CUDAMEMCPY(debug_out_cpu, out, double, ng, cudaMemcpyDeviceToHost);
}

void debug_operator_memcpy_post(double *out, double *buf, int ng, int ng2)
{
    GPAW_CUDAMEMCPY(debug_out_gpu, out, double, ng, cudaMemcpyDeviceToHost);
    GPAW_CUDAMEMCPY(debug_buf_gpu, buf, double, ng2, cudaMemcpyDeviceToHost);
}


void debug_operator_relax(OperatorObject* self, int relax_method, double w,
                          int n)
{
    MPI_Request recvreq[2];
    MPI_Request sendreq[2];
    int i;

    boundary_conditions* bc = self->bc;
    const int *size1 = bc->size1;
    const int *size2 = bc->size2;
    int ng = bc->ndouble * size1[0] * size1[1] * size1[2];
    int ng2 = bc->ndouble * size2[0] * size2[1] * size2[2];

    const double_complex *ph;
    ph = 0;

    for (i=0; i < 3; i++) {
        bc_unpack1(bc, debug_out_cpu, debug_buf_cpu, i, recvreq, sendreq,
                   debug_recvbuf, debug_sendbuf, ph + 2 * i, 0, 1);
        bc_unpack2(bc, debug_buf_cpu, i, recvreq, sendreq,
                   debug_recvbuf, 1);
    }
    bmgs_relax(relax_method, &self->stencil, debug_buf_gpu, debug_out_cpu,
               debug_in_cpu, w);

    double buf_err = 0;
    for (i=0; i < ng2; i++) {
        buf_err = MAX(buf_err, fabs(debug_buf_cpu[i] - debug_buf_gpu[i]));
    }
    double fun_err = 0;
    for (i=0; i < ng; i++) {
        fun_err = MAX(fun_err, fabs(debug_out_cpu[i] - debug_out_gpu[i]));
    }
    int rank = 0;
    if (bc->comm != MPI_COMM_NULL)
        MPI_Comm_rank(bc->comm, &rank);
    if (buf_err > GPAW_CUDA_ABS_TOL) {
        fprintf(stderr,
                "Debug cuda operator relax bc (n:%d rank:%d) errors: buf %g\n",
                n, rank, buf_err);
    }
    if (fun_err > GPAW_CUDA_ABS_TOL) {
        fprintf(stderr,
                "Debug cuda operator relax (n:%d rank:%d) errors: fun %g\n",
                n, rank, fun_err);
    }
}

void debug_operator_apply(OperatorObject* self, const double_complex *ph,
                          bool real, int n, int myblocks)
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

    for (i=0; i < 3; i++) {
        bc_unpack1(bc, debug_in_cpu, debug_buf_cpu, i, recvreq, sendreq,
                   debug_recvbuf, debug_sendbuf, ph + 2 * i, 0, myblocks);
        bc_unpack2(bc, debug_buf_cpu, i, recvreq, sendreq,
                   debug_recvbuf, myblocks);
    }
    for (int m=0; m < myblocks; m++) {
        if (real)
            bmgs_fd(&self->stencil, debug_buf_gpu + m * ng2,
                    debug_out_cpu + m * ng);
        else
            bmgs_fdz(&self->stencil,
                    (const double_complex*) (debug_buf_gpu + m * ng2),
                    (double_complex*) (debug_out_cpu + m * ng));
    }

    double buf_err = 0;
    int buf_err_n = 0;
    for (i = 0; i < ng2 * myblocks; i++) {
        err = fabs(debug_buf_cpu[i] - debug_buf_gpu[i]);
        if (err > GPAW_CUDA_ABS_TOL)
            buf_err_n++;
        buf_err = MAX(buf_err, err);
    }
    double out_err = 0;
    int out_err_n = 0;
    for (i = 0; i < ng * myblocks; i++) {
        err = fabs(debug_out_cpu[i] - debug_out_gpu[i]);
        if (err > GPAW_CUDA_ABS_TOL)
            out_err_n++;
        out_err = MAX(out_err, err);
    }
    int rank = 0;
    if (bc->comm != MPI_COMM_NULL)
        MPI_Comm_rank(bc->comm, &rank);
    if (buf_err > GPAW_CUDA_ABS_TOL) {
        printf("Debug cuda operator fd bc (n:%d rank:%d) errors: buf %g count %d/%d\n",
               n, rank, buf_err, buf_err_n, ng2 * myblocks);
        fflush(stdout);
    }
    if (out_err > GPAW_CUDA_ABS_TOL) {
        printf("Debug cuda operator fd (n:%d rank:%d) errors: out %g count %d/%d\n",
               n, rank, out_err, out_err_n, ng * myblocks);
        fflush(stdout);
    }
}

PyObject* Operator_relax_cuda_gpu(OperatorObject* self, PyObject* args)
{
    int relax_method;
    CUdeviceptr func_gpu;
    CUdeviceptr source_gpu;
    double w = 1.0;
    int nrelax;

    if (!PyArg_ParseTuple(args, "inni|d", &relax_method, &func_gpu,
                          &source_gpu, &nrelax, &w))
        return NULL;

    boundary_conditions* bc = self->bc;

    double *fun = (double*) func_gpu;
    const double *src = (double*) source_gpu;
    const double_complex *ph;

    const int *size2 = bc->size2;
    const int *size1 = bc->size1;
    int ng = bc->ndouble * size1[0] * size1[1] * size1[2];
    int ng2 = bc->ndouble * size2[0] * size2[1] * size2[2];

    MPI_Request recvreq[3][2];
    MPI_Request sendreq[3][2];

    if (gpaw_cuda_debug) {
        debug_operator_allocate(bc, ng, ng2, 1);
    }

    ph = 0;
    int blocks = 1;
    operator_alloc_buffers(self, blocks);

    int boundary = 0;
    if (bc->sendproc[0][0] != DO_NOTHING)
        boundary |= GPAW_BOUNDARY_X0;
    if (bc->sendproc[0][1] != DO_NOTHING)
        boundary |= GPAW_BOUNDARY_X1;
    if (bc->sendproc[1][0] != DO_NOTHING)
        boundary |= GPAW_BOUNDARY_Y0;
    if (bc->sendproc[1][1] != DO_NOTHING)
        boundary |= GPAW_BOUNDARY_Y1;
    if (bc->sendproc[2][0] != DO_NOTHING)
        boundary |= GPAW_BOUNDARY_Z0;
    if (bc->sendproc[2][1] != DO_NOTHING)
        boundary |= GPAW_BOUNDARY_Z1;

    int cuda_overlap = bmgs_fd_boundary_test(&self->stencil_gpu, boundary,
                                             bc->ndouble);
    int nsendrecvs = 0;
    for (int i=0; i < 3; i++) {
        for (int j=0; j < 2; j++) {
            nsendrecvs += MAX(bc->nsend[i][j], bc->nrecv[i][j])
                        * blocks * sizeof(double);
        }
    }
    cuda_overlap &= (nsendrecvs > GPAW_CUDA_OVERLAP_SIZE);
    if (cuda_overlap)
        cudaEventRecord(operator_event[1], 0);

    for (int n=0; n < nrelax; n++ ) {
        if (gpaw_cuda_debug) {
            debug_operator_memcpy_pre(fun, src, ng);
        }
        if (cuda_overlap) {
            cudaStreamWaitEvent(operator_stream[0], operator_event[1], 0);
            bc_unpack_paste_cuda_gpu(bc, fun, operator_buf_gpu, recvreq,
                                     operator_stream[0], 1);
            cudaEventRecord(operator_event[0], operator_stream[0]);

            bmgs_relax_cuda_gpu(relax_method, &self->stencil_gpu,
                                operator_buf_gpu, fun, src, w,
                                boundary|GPAW_BOUNDARY_SKIP,
                                operator_stream[0]);
            cudaStreamWaitEvent(operator_stream[1], operator_event[0], 0);
            for (int i=0; i < 3; i++) {
                bc_unpack_cuda_gpu_async(bc, fun, operator_buf_gpu, i,
                                         recvreq, sendreq[i], ph + 2 * i,
                                         operator_stream[1], 1);
            }
            bmgs_relax_cuda_gpu(relax_method, &self->stencil_gpu,
                                operator_buf_gpu, fun, src, w,
                                boundary|GPAW_BOUNDARY_ONLY,
                                operator_stream[1]);
            cudaEventRecord(operator_event[1], operator_stream[1]);
        } else {
            bc_unpack_paste_cuda_gpu(bc, fun, operator_buf_gpu, recvreq,
                                     0, 1);
            for (int i=0; i < 3; i++) {
                bc_unpack_cuda_gpu(bc, fun, operator_buf_gpu, i,
                                   recvreq, sendreq[i], ph + 2 * i, 0, 1);
            }
            bmgs_relax_cuda_gpu(relax_method, &self->stencil_gpu,
                                operator_buf_gpu, fun, src, w,
                                GPAW_BOUNDARY_NORMAL, 0);
        }

        if (gpaw_cuda_debug) {
            cudaDeviceSynchronize();
            debug_operator_memcpy_post(fun, operator_buf_gpu, ng, ng2);
            debug_operator_relax(self, relax_method, w, n);
        }
    }

    if (gpaw_cuda_debug) {
        debug_operator_deallocate();
    }

    if (cuda_overlap) {
        cudaStreamWaitEvent(0, operator_event[1], 0);
        cudaStreamSynchronize(operator_stream[0]);
    }
    if (PyErr_Occurred())
        return NULL;
    else
        Py_RETURN_NONE;
}

PyObject * Operator_apply_cuda_gpu(OperatorObject* self, PyObject* args)
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

    boundary_conditions* bc = self->bc;
    const int *size1 = bc->size1;
    const int *size2 = bc->size2;
    int ng = bc->ndouble * size1[0] * size1[1] * size1[2];
    int ng2 = bc->ndouble * size2[0] * size2[1] * size2[2];

    const double *in = (double*) input_gpu;
    double *out = (double*) output_gpu;
    const double_complex *ph;

    bool real = (type->type_num == NPY_DOUBLE);

    MPI_Request recvreq[3][2];
    MPI_Request sendreq[3][2];

    if (real)
        ph = 0;
    else
        ph = COMPLEXP(phases);

    int mpi_size = 1;
    if ((bc->maxsend || bc->maxrecv) && bc->comm != MPI_COMM_NULL) {
        MPI_Comm_size(bc->comm, &mpi_size);
    }
    int blocks = MAX(MIN(MIN(mpi_size * (GPAW_CUDA_BLOCKS_MIN),
                             (GPAW_CUDA_BLOCKS_MAX) / bc->ndouble),
                         nin),
                     1);

    if (gpaw_cuda_debug) {
        debug_operator_allocate(bc, ng, ng2, blocks);
    }

    operator_alloc_buffers(self, blocks);

    int boundary = 0;
    if (bc->sendproc[0][0] != DO_NOTHING)
        boundary |= GPAW_BOUNDARY_X0;
    if (bc->sendproc[0][1] != DO_NOTHING)
        boundary |= GPAW_BOUNDARY_X1;
    if (bc->sendproc[1][0] != DO_NOTHING)
        boundary |= GPAW_BOUNDARY_Y0;
    if (bc->sendproc[1][1] != DO_NOTHING)
        boundary |= GPAW_BOUNDARY_Y1;
    if (bc->sendproc[2][0] != DO_NOTHING)
        boundary |= GPAW_BOUNDARY_Z0;
    if (bc->sendproc[2][1] != DO_NOTHING)
        boundary |= GPAW_BOUNDARY_Z1;

    int cuda_overlap = bmgs_fd_boundary_test(&self->stencil_gpu, boundary,
                                             bc->ndouble);
    int nsendrecvs = 0;
    for (int i=0; i < 3; i++) {
        for (int j=0; j < 2; j++) {
            nsendrecvs += MAX(bc->nsend[i][j], bc->nrecv[i][j])
                        * blocks * sizeof(double);
        }
    }
    cuda_overlap &= (nsendrecvs > GPAW_CUDA_OVERLAP_SIZE);
    if  (cuda_overlap)
        cudaEventRecord(operator_event[1], 0);

    for (int n=0; n < nin; n += blocks) {
        const double *in2 = in + n * ng;
        double *out2 = out + n * ng;
        int myblocks = MIN(blocks, nin - n);
        if (gpaw_cuda_debug) {
            debug_operator_memcpy_pre(out2, in2, ng * myblocks);
        }
        if (cuda_overlap) {
            cudaStreamWaitEvent(operator_stream[0], operator_event[1], 0);
            bc_unpack_paste_cuda_gpu(bc, in2, operator_buf_gpu, recvreq,
                                     operator_stream[0], myblocks);
            cudaEventRecord(operator_event[0], operator_stream[0]);

            if (real) {
                bmgs_fd_cuda_gpu(&self->stencil_gpu, operator_buf_gpu, out2,
                                 boundary|GPAW_BOUNDARY_SKIP, myblocks,
                                 operator_stream[0]);
            } else {
                bmgs_fd_cuda_gpuz(&self->stencil_gpu,
                                  (const cuDoubleComplex*) operator_buf_gpu,
                                  (cuDoubleComplex*) out2,
                                  boundary|GPAW_BOUNDARY_SKIP, myblocks,
                                  operator_stream[0]);
            }
            cudaStreamWaitEvent(operator_stream[1], operator_event[0], 0);
            for (int i=0; i < 3; i++) {
                bc_unpack_cuda_gpu_async(bc, in2, operator_buf_gpu, i,
                                         recvreq, sendreq[i], ph + 2 * i,
                                         operator_stream[1], myblocks);
            }
            if (real) {
                bmgs_fd_cuda_gpu(&self->stencil_gpu, operator_buf_gpu, out2,
                                 boundary|GPAW_BOUNDARY_ONLY, myblocks,
                                 operator_stream[1]);
            } else {
                bmgs_fd_cuda_gpuz(&self->stencil_gpu,
                                  (const cuDoubleComplex*) operator_buf_gpu,
                                  (cuDoubleComplex*) out2,
                                  boundary|GPAW_BOUNDARY_ONLY, myblocks,
                                  operator_stream[1]);
            }
            cudaEventRecord(operator_event[1], operator_stream[1]);
        } else {
            bc_unpack_paste_cuda_gpu(bc, in2, operator_buf_gpu, recvreq,
                                     0, myblocks);
            for (int i=0; i < 3; i++) {
                bc_unpack_cuda_gpu(bc, in2, operator_buf_gpu, i,
                                   recvreq, sendreq[i], ph + 2 * i,
                                   0, myblocks);
            }
            if (real) {
                bmgs_fd_cuda_gpu(&self->stencil_gpu, operator_buf_gpu, out2,
                                 GPAW_BOUNDARY_NORMAL, myblocks, 0);
            } else {
                bmgs_fd_cuda_gpuz(&self->stencil_gpu,
                                  (const cuDoubleComplex*) (operator_buf_gpu),
                                  (cuDoubleComplex*) out2,
                                  GPAW_BOUNDARY_NORMAL, myblocks, 0);
            }
        }

        if (gpaw_cuda_debug) {
            cudaDeviceSynchronize();
            debug_operator_memcpy_post(out2, operator_buf_gpu,
                                       ng * myblocks, ng2 * myblocks);
            debug_operator_apply(self, ph, real, n, myblocks);
        }
    }

    if (gpaw_cuda_debug) {
        debug_operator_deallocate();
    }

    if (cuda_overlap) {
        cudaStreamWaitEvent(0, operator_event[1], 0);
        cudaStreamSynchronize(operator_stream[0]);
    }
    if (PyErr_Occurred())
        return NULL;
    else
        Py_RETURN_NONE;
}
