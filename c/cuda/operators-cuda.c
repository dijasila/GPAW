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



PyObject * Operator_relax_cuda_cpu(OperatorObject *self,
					  PyObject *args)
{
  
  int relax_method;
  PyArrayObject* func;
  PyArrayObject* source;
  double w = 1.0;
  int nrelax;
  if (!PyArg_ParseTuple(args, "iOOi|d", &relax_method, &func, &source, 
			&nrelax, &w))
    return NULL;
  
  const boundary_conditions* bc = self->bc;

  double* fun = DOUBLEP(func);
  const double* src = DOUBLEP(source);
  const double_complex* ph;

  const int* size2 = bc->size2;
  double* buf = GPAW_MALLOC(double, size2[0] * size2[1] * size2[2] *
                            bc->ndouble);
  double* sendbuf = GPAW_MALLOC(double, bc->maxsend);
  double* recvbuf = GPAW_MALLOC(double, bc->maxrecv);

  ph = 0;

  for (int n = 0; n < nrelax; n++ )
    {
      for (int i = 0; i < 3; i++)
        {
          bc_unpack1(bc, fun, buf, i,
		     self->recvreq, self->sendreq,
		     recvbuf, sendbuf, ph + 2 * i, 0, 1);
          bc_unpack2(bc, buf, i,
		     self->recvreq, self->sendreq, recvbuf, 1);
        }
      bmgs_relax_cuda_cpu(relax_method, &self->stencil, buf, fun, src, w);
    }
  free(recvbuf);
  free(sendbuf);
  free(buf);
  Py_RETURN_NONE;
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
  
  const boundary_conditions* bc = self->bc;

  double* fun = (double*)func_gpu;
  const double* src = (double*)source_gpu;
  const double_complex* ph;

  //  double* sendbuf = GPAW_MALLOC(double, bc->maxsend);
  //double* recvbuf = GPAW_MALLOC(double, bc->maxrecv);
  const int* size2 = bc->size2;
  const int* size1 = bc->size1;
  int ng = bc->ndouble * size1[0] * size1[1] * size1[2];
  ph = 0;

  int blocks=1;

  if (blocks>self->alloc_blocks){
    if (self->buf_gpu) cudaFree(self->buf_gpu);
    GPAW_CUDAMALLOC(&(self->buf_gpu), double, 
		    size2[0] * size2[1] * size2[2] 
		    * bc->ndouble *  blocks);
    if (self->sendbuf) cudaFreeHost(self->sendbuf);
    GPAW_CUDAMALLOC_HOST(&(self->sendbuf),double, bc->maxsend * blocks);
    if (self->recvbuf) cudaFreeHost(self->recvbuf);
    GPAW_CUDAMALLOC_HOST(&(self->recvbuf),double, bc->maxrecv * blocks);

    if (self->sendbuf_gpu) cudaFree(self->sendbuf_gpu);
    GPAW_CUDAMALLOC(&(self->sendbuf_gpu),double, bc->maxsend * blocks);
    if (self->recvbuf_gpu) cudaFree(self->recvbuf_gpu);
    GPAW_CUDAMALLOC(&(self->recvbuf_gpu),double, bc->maxrecv * blocks);

    self->alloc_blocks=blocks;
  }


  for (int n = 0; n < nrelax; n++ )
    {
      for (int i = 0; i < 3; i++)
        {
          bc_unpack1_cuda_gpu(bc, fun, self->buf_gpu, i,
			      self->recvreq, self->sendreq,
			      self->recvbuf, self->sendbuf,
			      self->sendbuf_gpu,
			      ph + 2 * i, 0, 1);
          bc_unpack2_cuda_gpu(bc, self->buf_gpu, i,
			      self->recvreq, self->sendreq, 
			      self->recvbuf,self->recvbuf_gpu,1);
	}
      bmgs_relax_cuda_gpu(relax_method, &self->stencil_gpu, self->buf_gpu, fun, src, w);
    }
  //free(recvbuf);
  //free(sendbuf);
  Py_RETURN_NONE;
}


/*
  intput->nd number of dimensions 
  input->dimensions[0]; An array of integers providing the shape in each dimension 
  input->descr->type_num A number that uniquely identifies the data type.
  DOUBLEP(input);  ((double*)((a)->data))
  DOUBLEP(output); ((double*)((a)->data))

 */


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

  if (real)
    ph = 0;
  else
    ph = COMPLEXP(phases);



  MPI_Request recvreq[2];
  MPI_Request sendreq[2];

  int blocks=MIN(GPAW_CUDA_BLOCKS,nin);
  
  //double* sendbuf = GPAW_MALLOC(double, bc->maxsend*blocks);
  //double* recvbuf = GPAW_MALLOC(double, bc->maxrecv*blocks);


  if (blocks>self->alloc_blocks){
    if (self->buf_gpu) cudaFree(self->buf_gpu);
    GPAW_CUDAMALLOC(&(self->buf_gpu), sizeof(double), 
		    size2[0] * size2[1] * size2[2] 
		    * bc->ndouble *  blocks);
    if (self->sendbuf) cudaFreeHost(self->sendbuf);
    GPAW_CUDAMALLOC_HOST(&(self->sendbuf),double, bc->maxsend * blocks);
    if (self->recvbuf) cudaFreeHost(self->recvbuf);
    GPAW_CUDAMALLOC_HOST(&(self->recvbuf),double, bc->maxrecv * blocks);

    if (self->sendbuf_gpu) cudaFree(self->sendbuf_gpu);
    GPAW_CUDAMALLOC(&(self->sendbuf_gpu),double, bc->maxsend * blocks);
    if (self->recvbuf_gpu) cudaFree(self->recvbuf_gpu);
    GPAW_CUDAMALLOC(&(self->recvbuf_gpu),double, bc->maxrecv * blocks);

    self->alloc_blocks=blocks;
  }
  double* buf = self->buf_gpu;
  
  for (int n = 0; n < nin; n+=blocks)
    {
      const double* in2 = in + n * ng;
      double* out2 = out + n * ng;
      int myblocks=MIN(blocks,nin-n);
      for (int i = 0; i < 3; i++) {
	  bc_unpack1_cuda_gpu(bc, in2, buf, i,
			      recvreq, sendreq,
			      self->recvbuf, self->sendbuf, 
			      self->sendbuf_gpu, 
			      ph + 2 * i, 0, myblocks);
	  bc_unpack2_cuda_gpu(bc, buf, i, recvreq, sendreq, 
			      self->recvbuf, self->recvbuf_gpu, myblocks);
      }
      if (real){
	bmgs_fd_cuda_gpu(&self->stencil_gpu, buf,out2,myblocks);
	//bmgs_fd_cuda_gpu_bc(&self->stencil_gpu, in2,out2,myblocks);
      }else{
	bmgs_fd_cuda_gpuz(&self->stencil_gpu, (const cuDoubleComplex*)buf,
			  (cuDoubleComplex*)out2,myblocks);
      }
    }    
  //free(recvbuf);
  //free(sendbuf);
  Py_RETURN_NONE;
}

struct apply_args{
  int thread_id;
  OperatorObject *self;
  int ng;
  int ng2;
  int nin;
  int nthds;
  int chunksize;
  int chunkinc;
  const double* in;
  double* out;
  int real;
  const double_complex* ph;
};

void *apply_worker_cuda_cpu(void *threadarg)
{
  struct apply_args *args = (struct apply_args *) threadarg;
  boundary_conditions* bc = args->self->bc;
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
  if (chunksize > args->chunksize)
    chunksize = args->chunksize;

  double* sendbuf = GPAW_MALLOC(double, bc->maxsend * args->chunksize);
  double* recvbuf = GPAW_MALLOC(double, bc->maxrecv * args->chunksize);
  double* buf = GPAW_MALLOC(double, args->ng2 * args->chunksize);

  for (int n = nstart; n < nend; n += chunksize)
    {
      if (n + chunksize >= nend && chunksize > 1)
        chunksize = nend - n;
      const double* in = args->in + n * args->ng;
      double* out = args->out + n * args->ng;
      for (int i = 0; i < 3; i++)
        {
          bc_unpack1(bc, in, buf, i,
                     recvreq, sendreq,
                     recvbuf, sendbuf, args->ph + 2 * i,
                     args->thread_id, chunksize);
          bc_unpack2(bc, buf, i, recvreq, sendreq, recvbuf, chunksize);
        }
      for (int m = 0; m < chunksize; m++)
        if (args->real)
          bmgs_fd_cuda_cpu(&args->self->stencil, buf + m * args->ng2, out + m * args->ng);
        else
	  assert(0);
      /*          bmgs_fdz(&args->self->stencil, (const double_complex*) (buf + m * args->ng2),
		  (double_complex*) (out + m * args->ng));*/
    }
  free(buf);
  free(recvbuf);
  free(sendbuf);
  return NULL;
}
/*
void *apply_worker_cuda_gpu(void *threadarg)
{
  struct apply_args *args = (struct apply_args *) threadarg;
  boundary_conditions* bc = args->self->bc;
  double* sendbuf = args->self->sendbuf;
  double* recvbuf = args->self->recvbuf;
  double* buf = args->self->buf_gpu;
  MPI_Request recvreq[2];
  MPI_Request sendreq[2];

  int chunksize = args->nin;
  if (!chunksize)
    chunksize = 1;
  int nstart = 0;
  if (nstart >= args->nin)
    return NULL;
  int nend = nstart + chunksize;
  if (nend > args->nin)
    nend = args->nin;
  if (chunksize > args->chunksize)
    chunksize = args->chunksize;

  for (int n = nstart; n < nend; n += chunksize)
    {
      if (n + chunksize >= nend && chunksize > 1)
        chunksize = nend - n;
      const double* in = args->in + n * args->ng;
      double* out = args->out + n * args->ng;
      for (int i = 0; i < 3; i++)
        {
          bc_unpack1_cuda_gpu(bc, in, buf, i,
                     recvreq, sendreq,
                     recvbuf, sendbuf, args->ph + 2 * i,
                     args->thread_id, chunksize);
          bc_unpack2_cuda_gpu(bc, buf, i, recvreq, sendreq, recvbuf, chunksize);
        }
      for (int m = 0; m < chunksize; m++)
        if (args->real)
          bmgs_fd_cuda_gpu(&args->self->stencil, buf + m * args->ng2, out + m * args->ng);
        else
	  assert(0);
    }
  return NULL;
}
*/

PyObject * Operator_apply_cuda_cpu(OperatorObject *self,
                                 PyObject *args)
{
  PyArrayObject* input;
  PyArrayObject* output;
  PyArrayObject* phases = 0;
  if (!PyArg_ParseTuple(args, "OO|O", &input, &output, &phases))
    return NULL;

  int nin = 1;
  if (input->nd == 4)
    nin = input->dimensions[0];

  boundary_conditions* bc = self->bc;
  const int* size1 = bc->size1;
  const int* size2 = bc->size2;
  int ng = bc->ndouble * size1[0] * size1[1] * size1[2];
  int ng2 = bc->ndouble * size2[0] * size2[1] * size2[2];

  const double* in = DOUBLEP(input);
  double* out = DOUBLEP(output);
  const double_complex* ph;

  bool real = (input->descr->type_num == PyArray_DOUBLE);

  //  fprintf(stdout,"Operator_apply_cuda_cpu\n");

  if (real)
    ph = 0;
  else
    ph = COMPLEXP(phases);

  int chunksize = 1;
  if (getenv("GPAW_CHUNK_SIZE") != NULL)
    chunksize = atoi(getenv("GPAW_CHUNK_SIZE"));

  int chunkinc = chunksize;
  if (getenv("GPAW_CHUNK_INC") != NULL)
    chunkinc = atoi(getenv("GPAW_CHUNK_INC"));

  int nthds = 1;
#ifdef GPAW_OMP
  if (getenv("OMP_NUM_THREADS") != NULL)
    nthds = atoi(getenv("OMP_NUM_THREADS"));
#endif
  struct apply_args *wargs = GPAW_MALLOC(struct apply_args, nthds);
  pthread_t *thds = GPAW_MALLOC(pthread_t, nthds);

  for(int i=0; i < nthds; i++)
    {
      (wargs+i)->thread_id = i;
      (wargs+i)->nthds = nthds;
      (wargs+i)->chunksize = chunksize;
      (wargs+i)->chunkinc = chunkinc;
      (wargs+i)->self = self;
      (wargs+i)->ng = ng;
      (wargs+i)->ng2 = ng2;
      (wargs+i)->nin = nin;
      (wargs+i)->in = in;
      (wargs+i)->out = out;
      (wargs+i)->real = real;
      (wargs+i)->ph = ph;
    }
  
  apply_worker_cuda_cpu(wargs);
#ifdef GPAW_OMP
  for(int i=1; i < nthds; i++)
    pthread_join(*(thds+i), NULL);
#endif
  free(wargs);
  free(thds);

  Py_RETURN_NONE;
}


