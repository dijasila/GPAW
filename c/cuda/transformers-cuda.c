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

  if (!PyArg_ParseTuple(args, "nnOO|O", &input_gpu, &output_gpu,&shape, &type,
			&phases))
    return NULL;

  int nin = 1;

  if (PyTuple_Size(shape)==4)
    nin = PyInt_AsLong(PyTuple_GetItem(shape,0));

  boundary_conditions* bc = self->bc;
  const int* size1 = bc->size1;

  int ng = bc->ndouble * size1[0] * size1[1] * size1[2];

  const double* in = (double*)input_gpu;
  double* out = (double*)output_gpu;

  const int* size2 = self->bc->size2;
  bool real = (type->type_num == PyArray_DOUBLE);
  const double_complex* ph = (real ? 0 : COMPLEXP(phases));

  MPI_Request recvreq[2];
  MPI_Request sendreq[2];

  int out_ng = bc->ndouble * self->size_out[0] * self->size_out[1]
               * self->size_out[2];

  int blocks=MIN(GPAW_CUDA_BLOCKS,nin);

  /*  
      double* sendbuf = GPAW_MALLOC(double, bc->maxsend * GPAW_ASYNC_D * blocks);
      double* recvbuf = GPAW_MALLOC(double, bc->maxrecv * GPAW_ASYNC_D * blocks);
  */
  if (blocks>self->alloc_blocks){
    if (self->buf_gpu) cudaFree(self->buf_gpu);
    GPAW_CUDAMALLOC(&(self->buf_gpu), double, 
		    size2[0] * size2[1] * size2[2] * 
		    self->bc->ndouble * blocks);

    if (self->buf2_gpu) cudaFree(self->buf2_gpu);
    GPAW_CUDAMALLOC(&(self->buf2_gpu), double,
		    16 * size2[0] * size2[1] * size2[2] * 
		    self->bc->ndouble);
    if (self->sendbuf) cudaFreeHost(self->sendbuf);
    GPAW_CUDAMALLOC_HOST(&(self->sendbuf),double, 
			 bc->maxsend * GPAW_ASYNC_D * blocks);
    if (self->recvbuf) cudaFreeHost(self->recvbuf);
    GPAW_CUDAMALLOC_HOST(&(self->recvbuf),double, 
			 bc->maxrecv * GPAW_ASYNC_D * blocks);

    if (self->sendbuf_gpu) cudaFree(self->sendbuf_gpu);
    GPAW_CUDAMALLOC(&(self->sendbuf_gpu),double, 
		    bc->maxsend * GPAW_ASYNC_D * blocks);
    if (self->recvbuf_gpu) cudaFree(self->recvbuf_gpu);
    GPAW_CUDAMALLOC(&(self->recvbuf_gpu),double, 
		    bc->maxrecv * GPAW_ASYNC_D * blocks);

    self->alloc_blocks=blocks;
  }
  double* buf = self->buf_gpu;
  double* buf2 = self->buf2_gpu;
  
  
  for (int n = 0; n < nin; n+=blocks)
    {
      const double* in2 = in + n * ng;
      double* out2 = out + n * out_ng;
      int myblocks=MIN(blocks,nin-n);
      for (int i = 0; i < 3; i++)
        {
          bc_unpack1_cuda_gpu(bc, in2, buf, i,
			      recvreq, sendreq,
			      self->recvbuf, self->sendbuf, 
			      self->sendbuf_gpu, 
			      ph + 2 * i, 0, myblocks);
          bc_unpack2_cuda_gpu(bc, buf, i,
			      recvreq, sendreq, 
			      self->recvbuf,self->recvbuf_gpu, myblocks);
        }
      for (int i = 0; i < myblocks; i++){
	
	if (real)
	  {
	    if (self->interpolate){
	      bmgs_interpolate_cuda_gpu(self->k, self->skip, buf+i*ng, 
					bc->size2,out2+i*out_ng, buf2);	      
	    }
	    else{
	      bmgs_restrict_cuda_gpu(self->k, buf+i*ng, bc->size2,
				     out2+i*out_ng, buf2);
	    }
	  }
	else
	  {
	    if (self->interpolate)
	      bmgs_interpolate_cuda_gpuz(self->k, self->skip, 
					 (cuDoubleComplex*)(buf+i*ng),
					 bc->size2, 
					 (cuDoubleComplex*)(out2+i*out_ng),
					 (cuDoubleComplex*) buf2);
	    else
	      bmgs_restrict_cuda_gpuz(self->k, 
				      (cuDoubleComplex*)(buf+i*ng),
				      bc->size2, 
				      (cuDoubleComplex*)(out2+i*out_ng),
				      (cuDoubleComplex*) buf2);
	  }
      }
    }
  //free(recvbuf);
  //free(sendbuf);
  Py_RETURN_NONE;
}

