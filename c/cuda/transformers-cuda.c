#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL GPAW_ARRAY_API
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>
#include <stdlib.h>
#include <pthread.h>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "../extensions.h"
#include "../transformers.h"
#include "gpaw-cuda.h"

PyObject* Transformer_apply_cuda_gpu(TransformerObject *self, PyObject *args)
{
  PyArrayObject* phases = 0;

  CUdeviceptr input_gpu;
  CUdeviceptr output_gpu;
  PyObject *shape;
  PyArray_Descr *type; 

  if (!PyArg_ParseTuple(args, "nnOO|O", &input_gpu, &output_gpu,&shape, &type, &phases))
    return NULL;

  int nin = 1;

  if (PyTuple_Size(shape)==4)
    nin = PyInt_AsLong(PyTuple_GetItem(shape,0));

  boundary_conditions* bc = self->bc;
  const int* size1 = bc->size1;

  int ng = bc->ndouble * size1[0] * size1[1] * size1[2];

  const double* in = (double*)input_gpu;
  double* out = (double*)output_gpu;

  bool real = (type->type_num == PyArray_DOUBLE);
  const double_complex* ph = (real ? 0 : COMPLEXP(phases));

  double* sendbuf = GPAW_MALLOC(double, bc->maxsend * GPAW_ASYNC_D);
  double* recvbuf = GPAW_MALLOC(double, bc->maxrecv * GPAW_ASYNC_D);
  double* buf = self->buf_gpu;
  double* buf2 = self->buf2_gpu;
  MPI_Request recvreq[2];
  MPI_Request sendreq[2];

  int out_ng = bc->ndouble * self->size_out[0] * self->size_out[1]
               * self->size_out[2];

  for (int n = 0; n < nin; n++)
    {
      const double* in2 = in + n * ng;
      double* out2 = out + n * out_ng;
      for (int i = 0; i < 3; i++)
        {
          bc_unpack1_cuda_gpu(bc, in2, buf, i,
                     recvreq, sendreq,
                     recvbuf, sendbuf, ph + 2 * i,
                     0, 1);
          bc_unpack2_cuda_gpu(bc, buf, i,
                     recvreq, sendreq, recvbuf, 1);
        }
      if (real)
        {
          if (self->interpolate)
            bmgs_interpolate_cuda_gpu(self->k, self->skip, buf, bc->size2,
                             out2, buf2);
          else
            bmgs_restrict_cuda_gpu(self->k, self->buf_gpu, bc->size2,
                          out2, buf2);
        }
      else
        {
          if (self->interpolate)
            bmgs_interpolate_cuda_gpuz(self->k, self->skip, (cuDoubleComplex*)buf,
				       bc->size2, (cuDoubleComplex*)out2,
				       (cuDoubleComplex*) buf2);
          else
            bmgs_restrict_cuda_gpuz(self->k, (cuDoubleComplex*) buf,
				    bc->size2, (cuDoubleComplex*)out2,
				    (cuDoubleComplex*) buf2);
        }
    }
  free(recvbuf);
  free(sendbuf);
  Py_RETURN_NONE;
}


/*
void *transapply_worker_cuda_gpu(void *threadarg)
{
  struct transapply_args *args = (struct transapply_args *) threadarg;
  boundary_conditions* bc = args->self->bc;
  TransformerObject *self = args->self;
  double* sendbuf = self->sendbuf + args->thread_id * bc->maxsend;
  double* recvbuf = self->recvbuf + args->thread_id * bc->maxrecv;
  double* buf = self->buf_gpu + args->thread_id * args->ng2;
  double* buf2 = self->buf2_gpu + args->thread_id * args->ng2 * 16;
  MPI_Request recvreq[2];
  MPI_Request sendreq[2];

  int chunksize = args->nin / args->nthds;
  if (!chunksize)
    chunksize = 1;
  int nstart = args->thread_id * chunksize;
  if (nstart >= args->nin)
    return NULL;
  int nend = nstart + chunksize;
  if (nend > args->nin)
    nend = args->nin;

  int out_ng;
  if (self->interpolate)
    out_ng = args->ng * 8;
  else
    out_ng = args->ng / 8;

  for (int n = nstart; n < nend; n++)
    {
      const double* in = args->in + n * args->ng;
      double* out = args->out + n * out_ng;
      for (int i = 0; i < 3; i++)
        {
          bc_unpack1_cuda_gpu(bc, in, buf, i,
                     recvreq, sendreq,
                     recvbuf, sendbuf, args->ph + 2 * i,
                     args->thread_id, 1);
          bc_unpack2_cuda_gpu(bc, buf, i,
                     recvreq, sendreq, recvbuf, 1);
        }
      if (args->real)
        {
          if (self->interpolate)
            bmgs_interpolate_cuda_gpu(self->k, self->skip, buf, bc->size2,
                             out, buf2);
          else
            bmgs_restrict_cuda_gpu(self->k, self->buf_gpu, bc->size2,
                          out, buf2);
        }
      else
        {
          if (self->interpolate)
            bmgs_interpolate_cuda_gpuz(self->k, self->skip, (cuDoubleComplex*)buf,
				       bc->size2, (cuDoubleComplex*)out,
				       (cuDoubleComplex*) buf2);
          else
            bmgs_restrict_cuda_gpuz(self->k, (cuDoubleComplex*) buf,
				    bc->size2, (cuDoubleComplex*)out,
				    (cuDoubleComplex*) buf2);
        }
    }
  return NULL;
}
*/
