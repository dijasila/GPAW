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

#ifdef DEBUG_CUDA
#define DEBUG_CUDA_TRANSFORMER
#endif //DEBUG_CUDA

static double *transformer_buf_gpu=NULL;
static int transformer_buf_size=0;
static int transformer_buf_max=0;
static int transformer_init_count=0;


void transformer_init_cuda(TransformerObject *self)
{
  const boundary_conditions* bc = self->bc;
  const int* size2 = bc->size2;  
  int ng2 = bc->ndouble * size2[0] * size2[1] * size2[2];
  
  transformer_buf_max=MAX(ng2,transformer_buf_max);

  transformer_init_count++;
}


void transformer_init_buffers(TransformerObject *self,int blocks)
{
  const boundary_conditions* bc = self->bc;
  const int* size2 = bc->size2;  
  int ng2 = (bc->ndouble * size2[0] * size2[1] * size2[2])*blocks;
  
  transformer_buf_max=MAX(ng2,transformer_buf_max);

  if (transformer_buf_max>transformer_buf_size){
    cudaFree(transformer_buf_gpu);
    cudaGetLastError();
    GPAW_CUDAMALLOC(&transformer_buf_gpu, double,transformer_buf_max);    
    transformer_buf_size=transformer_buf_max;
  }
}

void transformer_init_buffers_cuda()
{    
  transformer_buf_gpu=NULL;
  transformer_buf_size=0;
  //  transformer_buf_max=0;
  transformer_init_count=0;
}

void transformer_dealloc_cuda(int force)
{
  if (force)
    transformer_init_count=1;
  
  if (transformer_init_count==1) {
    cudaFree(transformer_buf_gpu);
    cudaGetLastError();
    transformer_init_buffers_cuda();
    return;
  }
  if (transformer_init_count>0) transformer_init_count--;
}


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
  const int* size2 = self->bc->size2;

  int ng = bc->ndouble * size1[0] * size1[1] * size1[2];
  int ng2 = bc->ndouble * size2[0] * size2[1] * size2[2];

  const double* in = (double*)input_gpu;
  double* out = (double*)output_gpu;

  bool real = (type->type_num == PyArray_DOUBLE);
  const double_complex* ph = (real ? 0 : COMPLEXP(phases));



  int out_ng = bc->ndouble * self->size_out[0] * self->size_out[1]
               * self->size_out[2];

  int mpi_size=1;
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size); 

  int blocks=MAX(MIN(MIN(mpi_size*(GPAW_CUDA_BLOCKS_MIN),
			 GPAW_CUDA_BLOCKS_MAX),nin),1);

  MPI_Request recvreq[3][2];
  MPI_Request sendreq[3][2];

#ifdef DEBUG_CUDA_TRANSFORMER
  MPI_Request recvreq_cpu[2];
  MPI_Request sendreq_cpu[2];

  double* sendbuf_cpu = GPAW_MALLOC(double, bc->maxsend * blocks);
  double* recvbuf_cpu = GPAW_MALLOC(double, bc->maxrecv * blocks);
  double* buf_cpu=GPAW_MALLOC(double, ng2 * blocks);
  double* buf2_cpu=GPAW_MALLOC(double, MAX(out_ng, ng2) * blocks);
  double* in_cpu=GPAW_MALLOC(double, ng * blocks);
  double* out_cpu=GPAW_MALLOC(double, out_ng * blocks);
  double* buf_cpu2=GPAW_MALLOC(double, ng2 * blocks);
  double* out_cpu2=GPAW_MALLOC(double, out_ng * blocks);  
#endif //DEBUG_CUDA_TRANSFORMER

  transformer_init_buffers(self,blocks);
  
  double* buf = transformer_buf_gpu;
  
  
  for (int n = 0; n < nin; n+=blocks)   {
    const double* in2 = in + n * ng;
    double* out2 = out + n * out_ng;
    int myblocks=MIN(blocks,nin-n);
#ifdef DEBUG_CUDA_TRANSFORMER
    GPAW_CUDAMEMCPY(in_cpu,in2,double, ng * myblocks, cudaMemcpyDeviceToHost); 
    GPAW_CUDAMEMCPY(out_cpu,out2,double, out_ng * myblocks, cudaMemcpyDeviceToHost);
#endif //DEBUG_CUDA_TRANSFORMER
    if (self->interpolate){    
      bc_unpack_paste_cuda_gpu(bc, in2, buf,
			       recvreq,
			       0, myblocks);
      for (int i = 0; i < 3; i++){
	bc_unpack_cuda_gpu(bc, in2, buf, i,
			   recvreq, sendreq[i],
			   ph+2*i, 0, myblocks);
      }
      /*
      bc_unpack_cuda_gpu_all(bc, in2, buf,recvreq,sendreq,
			     ph,0, myblocks);*/
#ifdef DEBUG_CUDA_TRANSFORMER
      GPAW_CUDAMEMCPY(buf_cpu2,transformer_buf_gpu,double, ng2 * myblocks, 
		    cudaMemcpyDeviceToHost);
#endif //DEBUG_CUDA_TRANSFORMER
      if (real)  {	
	bmgs_interpolate_cuda_gpu(self->k, self->skip, buf, 
				  bc->size2, out2, self->size_out,
				  myblocks);
      } else {
	bmgs_interpolate_cuda_gpuz(self->k, self->skip, 
				   (cuDoubleComplex*)(buf),
				   bc->size2, 
				   (cuDoubleComplex*)(out2),
				   self->size_out,
				   myblocks);
      }
    }else  {
      bc_unpack_paste_cuda_gpu(bc, in2, buf,
			       recvreq,
			       0, myblocks);
      for (int i = 0; i < 3; i++){
	bc_unpack_cuda_gpu(bc, in2, buf, i,
			   recvreq, sendreq[i],
			   ph+2*i, 0, myblocks);
      }
      /*
	bc_unpack_cuda_gpu_all(bc, in2, buf,recvreq,sendreq,
	ph,0, myblocks);*/
#ifdef DEBUG_CUDA_TRANSFORMER
      GPAW_CUDAMEMCPY(buf_cpu2,transformer_buf_gpu,double, ng2 * myblocks, 
		      cudaMemcpyDeviceToHost);
#endif //DEBUG_CUDA_TRANSFORMER
      if (real)  {
	bmgs_restrict_cuda_gpu(self->k, buf, bc->size2,
				out2, self->size_out, 
				myblocks);	
      } else {
	bmgs_restrict_cuda_gpuz(self->k, 
				 (cuDoubleComplex*)(buf),
				 bc->size2, 
				 (cuDoubleComplex*)(out2),
				 self->size_out,
				 myblocks);
      }
    }
    
#ifdef DEBUG_CUDA_TRANSFORMER
    for (int i = 0; i < 3; i++)   {
      bc_unpack1(bc, in_cpu, buf_cpu, i,
		 recvreq_cpu, sendreq_cpu,
		 recvbuf_cpu, sendbuf_cpu, ph + 2 * i, 0, myblocks);
      bc_unpack2(bc, buf_cpu, i,
		 recvreq_cpu, sendreq_cpu, recvbuf_cpu, myblocks);	  
    }
    double buf_err=0;
    for (int i=0;i<ng2*myblocks;i++) {      
      buf_err=MAX(buf_err,fabs(buf_cpu[i]-buf_cpu2[i]));
    }
    for (int m = 0; m < myblocks; m++){
      if (real)  {
	if (self->interpolate)
	  bmgs_interpolate(self->k, self->skip, buf_cpu2 + m * ng2, bc->size2,
			   out_cpu+m * out_ng, buf2_cpu + m * MAX(out_ng, ng2));
	else
	  bmgs_restrict(self->k, buf_cpu2 + m * ng2, bc->size2,
			out_cpu+m * out_ng,buf2_cpu + m * MAX(out_ng, ng2));
      }
      else  {
	if (self->interpolate)
	  bmgs_interpolatez(self->k, self->skip, 
			    (double_complex*)(buf_cpu2 + m * ng2),
			    bc->size2, (double_complex*)(out_cpu+m * out_ng),
			    (double_complex*) (buf2_cpu + m * MAX(out_ng, ng2)));
	else
	  bmgs_restrictz(self->k, (double_complex*) (buf_cpu2 + m * ng2),
			 bc->size2, (double_complex*)(out_cpu+m * out_ng),
			 (double_complex*) (buf2_cpu + m * MAX(out_ng, ng2)));
      }
    }
    GPAW_CUDAMEMCPY(out_cpu2,out2,double, out_ng * myblocks, 
		    cudaMemcpyDeviceToHost);    
    double out_err=0;
    for (int i=0;i<out_ng*myblocks;i++) {      
      out_err=MAX(out_err,fabs(out_cpu[i]-out_cpu2[i]));	
    }
    int rank=0;
    if (bc->comm != MPI_COMM_NULL)
    MPI_Comm_rank(bc->comm,&rank);      
    if (buf_err>GPAW_CUDA_ABS_TOL) {      
      fprintf(stderr,
	      "Debug cuda transformer bc (n:%d rank:%d) errors: buf %g\n",n,
	      rank,buf_err);
    } 
    if (out_err>GPAW_CUDA_ABS_TOL) {      
      fprintf(stderr,
	      "Debug cuda transformer (n:%d rank:%d) errors: out %g\n",n,
	      rank,out_err);
    }
#endif //DEBUG_CUDA_TRANSFORMER
  }

#ifdef DEBUG_CUDA_TRANSFORMER  
  free(sendbuf_cpu);
  free(recvbuf_cpu);
  free(buf_cpu);
  free(buf2_cpu);
  free(in_cpu);
  free(out_cpu);
  free(buf_cpu2);
  free(out_cpu2);
#endif //DEBUG_CUDA_TRANSFORMER

  if (PyErr_Occurred())
    return NULL;
  else
    Py_RETURN_NONE;
}

