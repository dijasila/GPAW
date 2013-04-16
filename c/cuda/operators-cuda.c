#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL GPAW_ARRAY_API
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>
#include <stdlib.h>
#include <pthread.h>

#include <cuda.h>
#include <cuda_runtime_api.h>


#include <../extensions.h>
#include <../operators.h>


#include "gpaw-cuda.h"


#ifdef DEBUG_CUDA
#define DEBUG_CUDA_OPERATOR 
#endif //DEBUG_CUDA

#define OPERATOR_NSTREAMS (2)

static cudaStream_t operator_stream[OPERATOR_NSTREAMS];
static cudaEvent_t operator_event[2];
static int operator_streams = 0;

static double *operator_buf_gpu=NULL;
static int operator_buf_size=0;
static int operator_buf_max=0;
static int operator_init_count=0;


void operator_init_cuda(OperatorObject *self)
{
  const boundary_conditions* bc = self->bc;
  const int* size2 = bc->size2;  
  int ng2 = bc->ndouble * size2[0] * size2[1] * size2[2];

  operator_buf_max=MAX(ng2,operator_buf_max);

  self->stencil_gpu = bmgs_stencil_to_gpu(&(self->stencil));
  operator_init_count++;
}


void operator_alloc_buffers(OperatorObject *self,int blocks)
{
  const boundary_conditions* bc = self->bc;
  const int* size2 = bc->size2;  
  int ng2 = (bc->ndouble * size2[0] * size2[1] * size2[2])*blocks;
  
  operator_buf_max=MAX(ng2,operator_buf_max);

  if (operator_buf_max>operator_buf_size){
    cudaFree(operator_buf_gpu);
    cudaGetLastError();
    GPAW_CUDAMALLOC(&operator_buf_gpu, double,operator_buf_max);    
    operator_buf_size=operator_buf_max;
  }
  if  (!operator_streams){
    for (int i=0;i<OPERATOR_NSTREAMS;i++){
      cudaStreamCreate(&(operator_stream[i]));
    }
    for (int i=0;i<2;i++){
      cudaEventCreateWithFlags(&operator_event[i],
			       cudaEventDefault|cudaEventDisableTiming);
    }

    operator_streams=OPERATOR_NSTREAMS;
  }
}

void operator_init_buffers_cuda()
{    
  operator_buf_gpu=NULL;
  operator_buf_size=0;
  //  operator_buf_max=0;
  operator_init_count=0;
  operator_streams=0;
}

void operator_dealloc_cuda(int force)
{
  if (force || (operator_init_count==1)) {
    cudaFree(operator_buf_gpu);
    if (operator_streams){
      for (int i=0;i<OPERATOR_NSTREAMS;i++){
	cudaStreamDestroy(operator_stream[i]);
      }
      for (int i=0;i<2;i++){
	cudaEventDestroy(operator_event[i]);  
      }      
    }
    cudaGetLastError();
    operator_init_buffers_cuda();
    return;
  }
  if (operator_init_count>0) operator_init_count--;
}

PyObject * Operator_relax_cuda_gpu(OperatorObject *self,
				   PyObject *args)
{
  int relax_method;
  CUdeviceptr func_gpu;
  CUdeviceptr source_gpu;

  double w = 1.0;
  int nrelax;
  if (!PyArg_ParseTuple(args, "inni|d", &relax_method, &func_gpu, &source_gpu, &nrelax, &w))
    return NULL;
  
  boundary_conditions* bc = self->bc;

  double* fun = (double*)func_gpu;
  const double* src = (double*)source_gpu;
  const double_complex* ph;

  const int* size2 = bc->size2;
  const int* size1 = bc->size1;
  int ng = bc->ndouble * size1[0] * size1[1] * size1[2];
  int ng2 = bc->ndouble * size2[0] * size2[1] * size2[2];

  MPI_Request recvreq[3][2];
  MPI_Request sendreq[3][2];


#ifdef DEBUG_CUDA_OPERATOR
  MPI_Request recvreq_cpu[2];
  MPI_Request sendreq_cpu[2];

  double* sendbuf_cpu = GPAW_MALLOC(double, bc->maxsend);
  double* recvbuf_cpu = GPAW_MALLOC(double, bc->maxrecv);
  double* buf_cpu=GPAW_MALLOC(double, ng2);
  double* fun_cpu=GPAW_MALLOC(double, ng);
  double* src_cpu=GPAW_MALLOC(double, ng);

  double* buf_cpu2=GPAW_MALLOC(double, ng2);
  double* fun_cpu2=GPAW_MALLOC(double, ng);
  
#endif //DEBUG_CUDA_OPERATOR

  ph = 0;

  int blocks=1;
	      
  operator_alloc_buffers(self,blocks);

  int boundary=0;
  
  if (bc->sendproc[0][0]!= DO_NOTHING)
    boundary|=GPAW_BOUNDARY_X0;
  if (bc->sendproc[0][1]!= DO_NOTHING)
    boundary|=GPAW_BOUNDARY_X1;
  if (bc->sendproc[1][0]!= DO_NOTHING)
    boundary|=GPAW_BOUNDARY_Y0;
  if (bc->sendproc[1][1]!= DO_NOTHING)
    boundary|=GPAW_BOUNDARY_Y1;
  if (bc->sendproc[2][0]!= DO_NOTHING)
    boundary|=GPAW_BOUNDARY_Z0;
  if (bc->sendproc[2][1]!= DO_NOTHING)
    boundary|=GPAW_BOUNDARY_Z1;

  int cuda_overlap=bmgs_fd_boundary_test(&self->stencil_gpu,boundary);

  int nsendrecvs=0;
  for (int i=0;i<3;i++)
    for (int j=0;j<2;j++)
      nsendrecvs+=MAX(bc->nsend[i][j],bc->nrecv[i][j])*blocks*sizeof(double);

  cuda_overlap&=(nsendrecvs>GPAW_CUDA_OVERLAP_SIZE);  
  
  if  (cuda_overlap) 
    cudaEventRecord(operator_event[1], 0);
  
  for (int n = 0; n < nrelax; n++ )    {
#ifdef DEBUG_CUDA_OPERATOR
    GPAW_CUDAMEMCPY(fun_cpu,fun,double, ng, cudaMemcpyDeviceToHost);
    GPAW_CUDAMEMCPY(src_cpu,src,double, ng, cudaMemcpyDeviceToHost);
#endif //DEBUG_CUDA_OPERATOR
    if  (cuda_overlap) {
      cudaStreamWaitEvent(operator_stream[0],operator_event[1],0);
      bc_unpack_paste_cuda_gpu(bc, fun, operator_buf_gpu,
			       recvreq,
			       operator_stream[0], 1);
      cudaEventRecord(operator_event[0], operator_stream[0]);
      bmgs_relax_cuda_gpu(relax_method, &self->stencil_gpu, operator_buf_gpu, 
			  fun, src, w,boundary|GPAW_BOUNDARY_SKIP,
			  operator_stream[0]);
      cudaStreamWaitEvent(operator_stream[1],operator_event[0],0);
      for (int i = 0; i < 3; i++){
	bc_unpack_cuda_gpu_async(bc, fun, operator_buf_gpu, i,
				 recvreq, sendreq[i],
				 ph+2*i, operator_stream[1],1);
      }	
      bmgs_relax_cuda_gpu(relax_method, &self->stencil_gpu, operator_buf_gpu, 
			  fun, src, w,boundary|GPAW_BOUNDARY_ONLY,
			  operator_stream[1]); 
      cudaEventRecord(operator_event[1], operator_stream[1]);
    }else{
      bc_unpack_paste_cuda_gpu(bc, fun, operator_buf_gpu,recvreq,
			       0, 1);
      for (int i = 0; i < 3; i++){
	bc_unpack_cuda_gpu(bc, fun, operator_buf_gpu, i,
			   recvreq, sendreq[i],
			   ph+2*i, 0,1);
      }
      bmgs_relax_cuda_gpu(relax_method, &self->stencil_gpu, operator_buf_gpu, 
			  fun, src, w,GPAW_BOUNDARY_NORMAL,0);
    } 
#ifdef DEBUG_CUDA_OPERATOR
    for (int i = 0; i < 3; i++)   {
      bc_unpack1(bc, fun_cpu, buf_cpu, i,
		 recvreq_cpu, sendreq_cpu,
		 recvbuf_cpu, sendbuf_cpu, ph + 2 * i, 0, 1);
      bc_unpack2(bc, buf_cpu, i,
		 recvreq_cpu, sendreq_cpu, recvbuf_cpu, 1);	  
    }
    GPAW_CUDAMEMCPY(buf_cpu2,operator_buf_gpu,double, ng2, cudaMemcpyDeviceToHost);
    bmgs_relax(relax_method, &self->stencil, buf_cpu2, fun_cpu, src_cpu, w);
    cudaDeviceSynchronize();
    GPAW_CUDAMEMCPY(fun_cpu2,fun,double, ng, cudaMemcpyDeviceToHost);    

    double fun_err=0;
    double buf_err=0;
    for (int i=0;i<ng2;i++) {      
      buf_err=MAX(buf_err,fabs(buf_cpu[i]-buf_cpu2[i]));
      if (i<ng){
	fun_err=MAX(fun_err,fabs(fun_cpu[i]-fun_cpu2[i]));
      }
    } 
    int rank=0;
    if (bc->comm != MPI_COMM_NULL)
      MPI_Comm_rank(bc->comm,&rank);
    if (buf_err>GPAW_CUDA_ABS_TOL) {
      fprintf(stderr,
	      "Debug cuda operator relax bc (n:%d rank:%d) errors: buf %g\n",
	      n, rank,buf_err);
    }
    if (fun_err>GPAW_CUDA_ABS_TOL) {
      fprintf(stderr,
	      "Debug cuda operator relax (n:%d rank:%d) errors: fun %g\n",
	      n,rank,fun_err);
    }
    
#endif //DEBUG_CUDA_OPERATOR
      
  }
#ifdef DEBUG_CUDA_OPERATOR

  free(sendbuf_cpu);
  free(recvbuf_cpu);
  free(buf_cpu);
  free(fun_cpu);
  free(src_cpu);
  free(buf_cpu2);
  free(fun_cpu2);
  
#endif //DEBUG_CUDA_OPERATOR
  if (cuda_overlap) {
    cudaStreamWaitEvent(0,operator_event[1],0);
    cudaStreamSynchronize(operator_stream[0]);      
  }
  if (PyErr_Occurred())
    return NULL;
  else
    Py_RETURN_NONE;
}


PyObject * Operator_apply_cuda_gpu(OperatorObject *self,
					  PyObject *args)
{
  PyArrayObject* phases = 0;

  CUdeviceptr input_gpu;
  CUdeviceptr output_gpu;
  PyObject *shape;
  PyArray_Descr *type; 

  if (!PyArg_ParseTuple(args, "nnOO|O", &input_gpu, &output_gpu, &shape, &type, &phases))
    return NULL;

  int nin = 1;

  if (PyTuple_Size(shape)==4)
    nin = PyInt_AsLong(PyTuple_GetItem(shape,0));
  
  boundary_conditions* bc = self->bc;
  const int* size1 = bc->size1;
  const int* size2 = bc->size2;
  int ng = bc->ndouble * size1[0] * size1[1] * size1[2];
  int ng2 = bc->ndouble * size2[0] * size2[1] * size2[2];

  const double* in = (double*)input_gpu;
  double* out = (double*)output_gpu;
  const double_complex* ph;
  
  bool real = (type->type_num == PyArray_DOUBLE);

  MPI_Request recvreq[3][2];
  MPI_Request sendreq[3][2];

  if (real)
    ph = 0;
  else
    ph = COMPLEXP(phases);

  int mpi_size=1;
  if (bc->comm != MPI_COMM_NULL)
    MPI_Comm_size(bc->comm, &mpi_size); 
  int blocks=MAX(MIN(MIN(mpi_size*(GPAW_CUDA_BLOCKS_MIN),GPAW_CUDA_BLOCKS_MAX),
		     nin),1);

#ifdef DEBUG_CUDA_OPERATOR
  MPI_Request recvreq_cpu[2];
  MPI_Request sendreq_cpu[2];

  double* sendbuf_cpu = GPAW_MALLOC(double, bc->maxsend * blocks);
  double* recvbuf_cpu = GPAW_MALLOC(double, bc->maxrecv * blocks);
  double* buf_cpu=GPAW_MALLOC(double, ng2 * blocks);
  double* in_cpu=GPAW_MALLOC(double, ng * blocks);
  double* out_cpu=GPAW_MALLOC(double, ng * blocks);
  double* buf_cpu2=GPAW_MALLOC(double, ng2 * blocks);
  double* out_cpu2=GPAW_MALLOC(double, ng * blocks);  
#endif //DEBUG_CUDA_OPERATOR
  
  operator_alloc_buffers(self,blocks);

  int boundary=0;
  
  if (bc->sendproc[0][0]!= DO_NOTHING)
    boundary|=GPAW_BOUNDARY_X0;
  if (bc->sendproc[0][1]!= DO_NOTHING)
    boundary|=GPAW_BOUNDARY_X1;
  if (bc->sendproc[1][0]!= DO_NOTHING)
    boundary|=GPAW_BOUNDARY_Y0;
  if (bc->sendproc[1][1]!= DO_NOTHING)
    boundary|=GPAW_BOUNDARY_Y1;
  if (bc->sendproc[2][0]!= DO_NOTHING)
    boundary|=GPAW_BOUNDARY_Z0;
  if (bc->sendproc[2][1]!= DO_NOTHING)
    boundary|=GPAW_BOUNDARY_Z1;  

  int cuda_overlap=bmgs_fd_boundary_test(&self->stencil_gpu,boundary);

  int nsendrecvs=0;
  for (int i=0;i<3;i++)
    for (int j=0;j<2;j++)
      nsendrecvs+=MAX(bc->nsend[i][j],bc->nrecv[i][j])*blocks*sizeof(double);
  
  cuda_overlap&=(nsendrecvs>GPAW_CUDA_OVERLAP_SIZE);

  if  (cuda_overlap) 
    cudaEventRecord(operator_event[1], 0);
  
  for (int n = 0; n < nin; n+=blocks)    {
    const double* in2 = in + n * ng;
    double* out2 = out + n * ng;
    int myblocks=MIN(blocks,nin-n);
#ifdef DEBUG_CUDA_OPERATOR
    GPAW_CUDAMEMCPY(in_cpu,in2,double, ng * myblocks, cudaMemcpyDeviceToHost); 
    GPAW_CUDAMEMCPY(out_cpu,out2,double, ng * myblocks, cudaMemcpyDeviceToHost);
#endif //DEBUG_CUDA_OPERATOR
    if (cuda_overlap) {      
      //printf("fd split\n");
      cudaStreamWaitEvent(operator_stream[0],operator_event[1],0);
      bc_unpack_paste_cuda_gpu(bc, in2, operator_buf_gpu,
			       recvreq,
			       operator_stream[0], myblocks);
      cudaEventRecord(operator_event[0], operator_stream[0]);
      if (real){
	bmgs_fd_cuda_gpu(&self->stencil_gpu, operator_buf_gpu,out2,
			 boundary|GPAW_BOUNDARY_SKIP,
			 myblocks,
			 operator_stream[0]);
      } else {
	bmgs_fd_cuda_gpuz(&self->stencil_gpu, 
			  (const cuDoubleComplex*)operator_buf_gpu,
			  (cuDoubleComplex*)out2,boundary|GPAW_BOUNDARY_SKIP,
			  myblocks,
			  operator_stream[0]);
      }
      cudaStreamWaitEvent(operator_stream[1],operator_event[0],0);
      for (int i = 0; i < 3; i++){
	bc_unpack_cuda_gpu_async(bc, in2, operator_buf_gpu, i,
				 recvreq, sendreq[i],
				 ph+2*i, operator_stream[1],  myblocks);
	
      }	
      if (real){
	bmgs_fd_cuda_gpu(&self->stencil_gpu, operator_buf_gpu,out2,
			 boundary|GPAW_BOUNDARY_ONLY,
			 myblocks,operator_stream[1]);
      }else{
	bmgs_fd_cuda_gpuz(&self->stencil_gpu, 
			  (const cuDoubleComplex*)operator_buf_gpu,
			  (cuDoubleComplex*)out2,
			  boundary|GPAW_BOUNDARY_ONLY,
			  myblocks,operator_stream[1]);
      }
      cudaEventRecord(operator_event[1], operator_stream[1]);
    }else{
      bc_unpack_paste_cuda_gpu(bc, in2, operator_buf_gpu,
			       recvreq,
			       0, myblocks);
      for (int i = 0; i < 3; i++){
	bc_unpack_cuda_gpu(bc, in2, operator_buf_gpu, i,
			   recvreq, sendreq[i],
			   ph+2*i, 0, myblocks);
      }
      if (real)
	bmgs_fd_cuda_gpu(&self->stencil_gpu, operator_buf_gpu,out2,
			 GPAW_BOUNDARY_NORMAL,
			 myblocks,0);
      else
	bmgs_fd_cuda_gpuz(&self->stencil_gpu, 
			  (const cuDoubleComplex*)(operator_buf_gpu),
			  (cuDoubleComplex*)out2,GPAW_BOUNDARY_NORMAL,myblocks,
			  0);
      
    }
    
#ifdef DEBUG_CUDA_OPERATOR
    for (int i = 0; i < 3; i++)   {
      bc_unpack1(bc, in_cpu, buf_cpu, i,
		 recvreq_cpu, sendreq_cpu,
		 recvbuf_cpu, sendbuf_cpu, ph + 2 * i, 0, myblocks);
      bc_unpack2(bc, buf_cpu, i,
		 recvreq_cpu, sendreq_cpu, recvbuf_cpu, myblocks);	  
    }
    GPAW_CUDAMEMCPY(buf_cpu2,operator_buf_gpu,double, ng2 * myblocks, 
		    cudaMemcpyDeviceToHost);
    for (int m = 0; m < myblocks; m++)
      if (real)
	bmgs_fd(&self->stencil, buf_cpu2 + m * ng2, out_cpu + m * ng);
      else
	bmgs_fdz(&self->stencil, (const double_complex*) (buf_cpu2 + m * ng2),
		 (double_complex*) (out_cpu + m * ng));
    cudaDeviceSynchronize();
    GPAW_CUDAMEMCPY(out_cpu2,out2,double, ng * myblocks, 
		    cudaMemcpyDeviceToHost);    
    double out_err=0;
    int out_err_n=0;
    double buf_err=0;
    int buf_err_n=0;
    for (int i=0;i<ng2*myblocks;i++) { 
      double err=fabs(buf_cpu[i]-buf_cpu2[i]);
      if (err>GPAW_CUDA_ABS_TOL) buf_err_n++;
      buf_err=MAX(buf_err,err);
      if (i<ng*myblocks){
	err=fabs(out_cpu[i]-out_cpu2[i]);
	if (err>GPAW_CUDA_ABS_TOL) out_err_n++;
	out_err=MAX(out_err,err);
      }
    }
    int rank=0;
    if (bc->comm != MPI_COMM_NULL)
      MPI_Comm_rank(bc->comm,&rank);      
    if (buf_err>GPAW_CUDA_ABS_TOL) {      
      printf("Debug cuda operator fd bc (n:%d rank:%d) errors: buf %g count %d/%d\n",n, rank,buf_err,buf_err_n,ng2*myblocks);
      fflush(stdout);
    } 
    if (out_err>GPAW_CUDA_ABS_TOL) {      
      printf("Debug cuda operator fd (n:%d rank:%d) errors: out %g count %d/%d\n",n,rank,out_err,out_err_n,ng*myblocks);
      fflush(stdout);
    }
#endif //DEBUG_CUDA_OPERATOR
  }    
#ifdef DEBUG_CUDA_OPERATOR
  
  free(sendbuf_cpu);
  free(recvbuf_cpu);
  free(buf_cpu);
  free(in_cpu);
  free(out_cpu);
  free(buf_cpu2);
  free(out_cpu2);
  
#endif //DEBUG_CUDA_OPERATOR
  if (cuda_overlap) {
    cudaStreamWaitEvent(0,operator_event[1],0);
    cudaStreamSynchronize(operator_stream[0]);      
  }

  if (PyErr_Occurred())
    return NULL;
  else
    Py_RETURN_NONE;
}




