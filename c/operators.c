/*  Copyright (C) 2003-2007  CAMP
 *  Copyright (C) 2007-2008  CAMd
 *  Copyright (C) 2005-2012  CSC - IT Center for Science Ltd.
 *  Please see the accompanying LICENSE file for further information. */

#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL GPAW_ARRAY_API
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>
#include <stdlib.h>
#include "extensions.h"
#include "bc.h"
#include "mympi.h"
#ifdef _OPENMP
#include <omp.h>
#endif
#include "threading.h"

#define __OPERATORS_C
#include "operators.h"
#undef __OPERATORS_C

#ifdef GPAW_ASYNC
  #define GPAW_ASYNC3 3
  #define GPAW_ASYNC2 2
#else
  #define GPAW_ASYNC3 1
  #define GPAW_ASYNC2 1
#endif


static void Operator_dealloc(OperatorObject *self)
{
  free(self->bc);
  PyObject_DEL(self);
}


static PyObject * Operator_relax(OperatorObject *self,
                                 PyObject *args)
{
  int relax_method;
  PyArrayObject* func;
  PyArrayObject* source;
  int nrelax;
  double w = 1.0;
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
      bmgs_relax(relax_method, &self->stencil, buf, fun, src, w);
    }
  free(recvbuf);
  free(sendbuf);
  free(buf);
  Py_RETURN_NONE;
}

/* The actual computation routine for finite difference operation
   Separating this routine helps using the same code in 
   C-preconditioner */
void apply_worker(OperatorObject *self, int chunksize, int start,
		  int end, int thread_id, int nthreads,
		  const double* in, double* out,
		  int real, const double_complex* ph)
{
  boundary_conditions* bc = self->bc;
  const int* size1 = bc->size1;
  const int* size2 = bc->size2;
  int ng = bc->ndouble * size1[0] * size1[1] * size1[2];
  int ng2 = bc->ndouble * size2[0] * size2[1] * size2[2];
  
  MPI_Request recvreq[2];
  MPI_Request sendreq[2];

  double* sendbuf = GPAW_MALLOC(double, bc->maxsend * chunksize);
  double* recvbuf = GPAW_MALLOC(double, bc->maxrecv * chunksize);
  double* buf = GPAW_MALLOC(double, ng2 * chunksize);

  const double* my_in;
  double* my_out;

  int nin = end - start;
  int nend = start + (nin / chunksize) * chunksize;
  int nremain = end - nend;

  for (int n = start; n < nend; n += chunksize)
    {
      my_in = in + n * ng;
      my_out = out + n * ng;

      for (int i = 0; i < 3; i++)
        {
          bc_unpack1(bc, my_in, buf, i,
                     recvreq, sendreq,
                     recvbuf, sendbuf, ph + 2 * i,
                     thread_id, chunksize);
          bc_unpack2(bc, buf, i, recvreq, sendreq, recvbuf, chunksize);
        }

// #pragma omp parallel for
      for (int m = 0; m < chunksize; m++)
        if (real)
          bmgs_fd(&self->stencil, buf + m * ng2, my_out + m * ng);
        else
          bmgs_fdz(&self->stencil, (const double_complex*) (buf + m * ng2),
                                         (double_complex*) (my_out + m * ng));
    }
  // Remainder 
  if (nremain > 0)
    {
      my_in = in + nend * ng;
      my_out = out + nend * ng;

      for (int i = 0; i < 3; i++)
        {
          bc_unpack1(bc, my_in, buf, i,
                     recvreq, sendreq,
                     recvbuf, sendbuf, ph + 2 * i,
                     thread_id, nremain);
          bc_unpack2(bc, buf, i, recvreq, sendreq, recvbuf, nremain);
        }

// #pragma omp parallel for
      for (int m = 0; m < nremain; m++)
        if (real)
          bmgs_fd(&self->stencil, buf + m * ng2, my_out + m * ng);
        else
          bmgs_fdz(&self->stencil, (const double_complex*) (buf + m * ng2),
                                         (double_complex*) (my_out + m * ng));
    }
  free(buf);
  free(recvbuf);
  free(sendbuf);
}


static PyObject * Operator_apply(OperatorObject *self,
                                 PyObject *args)
{
  PyArrayObject* input;
  PyArrayObject* output;
  PyArrayObject* phases = 0;
  if (!PyArg_ParseTuple(args, "OO|O", &input, &output, &phases))
    return NULL;

  int nin = 1;
  if (PyArray_NDIM(input) == 4)
    nin = PyArray_DIMS(input)[0];

  const double* in = DOUBLEP(input);
  double* out = DOUBLEP(output);

  bool real = (PyArray_DESCR(input)->type_num == NPY_DOUBLE);

  const double_complex* ph;
  if (real)
    ph = 0;
  else
    ph = COMPLEXP(phases);

  int chunksize = 1;
  boundary_conditions* bc = self->bc;
  if (getenv("GPAW_MPI_OPTIMAL_MSG_SIZE") != NULL)
    {
      int opt_msg_size = atoi(getenv("GPAW_MPI_OPTIMAL_MSG_SIZE"));
      if (bc->maxsend > 0 )
          chunksize = opt_msg_size * 1024 / (bc->maxsend / 2 * (2 - (int)real) *
                                             sizeof(double));
      chunksize = (chunksize > 0) ? chunksize : 1;
      chunksize = (chunksize < nin) ? chunksize : nin;
      // printf("Chunksize: %d maxsend: %d\n", chunksize, bc->maxsend);
    }
  
//#pragma omp parallel
{
  int thread_id = 0;
  int nthreads = 1;
  int start, end;
// #ifdef _OPENMP
//  thread_id = omp_get_thread_num();
//  nthreads = omp_get_num_threads();
// #endif
  SHARE_WORK(nin, nthreads, thread_id, &start, &end);
  apply_worker(self, chunksize, start, end, thread_id, nthreads,
	       in, out, real, ph);

} // end #omp parallel

  Py_RETURN_NONE;
}


static PyObject * Operator_get_diagonal_element(OperatorObject *self,
                                              PyObject *args)
{
  if (!PyArg_ParseTuple(args, ""))
    return NULL;

  const bmgsstencil* s = &self->stencil;
  double d = 0.0;
  for (int n = 0; n < s->ncoefs; n++)
    if (s->offsets[n] == 0)
      d = s->coefs[n];

  return Py_BuildValue("d", d);
}

static PyObject * Operator_get_async_sizes(OperatorObject *self, PyObject *args)
{
  if (!PyArg_ParseTuple(args, ""))
    return NULL;

#ifdef GPAW_ASYNC
  return Py_BuildValue("(iii)", 1, GPAW_ASYNC2, GPAW_ASYNC3);
#else
  return Py_BuildValue("(iii)", 0, GPAW_ASYNC2, GPAW_ASYNC3);
#endif
}

static PyMethodDef Operator_Methods[] = {
    {"apply",
     (PyCFunction)Operator_apply, METH_VARARGS, NULL},
    {"relax",
     (PyCFunction)Operator_relax, METH_VARARGS, NULL},
    {"get_diagonal_element",
     (PyCFunction)Operator_get_diagonal_element, METH_VARARGS, NULL},
    {"get_async_sizes",
     (PyCFunction)Operator_get_async_sizes, METH_VARARGS, NULL},
    {NULL, NULL, 0, NULL}

};


static PyObject* Operator_getattr(PyObject *obj, char *name)
{
    return Py_FindMethod(Operator_Methods, obj, name);
}

static PyTypeObject OperatorType = {
  PyObject_HEAD_INIT(&PyType_Type)
  0,
  "Operator",
  sizeof(OperatorObject),
  0,
  (destructor)Operator_dealloc,
  0,
  Operator_getattr
};

PyObject * NewOperatorObject(PyObject *obj, PyObject *args)
{
  PyArrayObject* coefs;
  PyArrayObject* offsets;
  PyArrayObject* size;
  int range;
  PyArrayObject* neighbors;
  int real;
  PyObject* comm_obj;
  int cfd;
  if (!PyArg_ParseTuple(args, "OOOiOiOi",
                        &coefs, &offsets, &size, &range, &neighbors,
                        &real, &comm_obj, &cfd))
    return NULL;

  OperatorObject *self = PyObject_NEW(OperatorObject, &OperatorType);
  if (self == NULL)
    return NULL;

  self->stencil = bmgs_stencil(PyArray_DIMS(coefs)[0], DOUBLEP(coefs),
                               LONGP(offsets), range, LONGP(size));

  const long (*nb)[2] = (const long (*)[2])LONGP(neighbors);
  const long padding[3][2] = {{range, range},
                             {range, range},
                             {range, range}};

  MPI_Comm comm = MPI_COMM_NULL;
  if (comm_obj != Py_None)
    comm = ((MPIObject*)comm_obj)->comm;

  int nthreads = 1;
#ifdef _OPENMP
  #pragma omp parallel
  {
    nthreads = omp_get_num_threads();
  }
#endif
  self->nthreads = nthreads;

  self->bc = bc_init(LONGP(size), padding, padding, nb, comm, real, cfd);
  return (PyObject*)self;
}
