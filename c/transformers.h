#ifndef TRANSFORMERS_H
#define TRANSFORMERS_H

#include "bc.h"

#ifdef GPAW_ASYNC
  #define GPAW_ASYNC_D 3
#else
  #define GPAW_ASYNC_D 1
#endif

typedef struct
{
  PyObject_HEAD
  boundary_conditions* bc;
  int p;
  int k;
  bool interpolate;
  MPI_Request recvreq[2];
  MPI_Request sendreq[2];
  int skip[3][2];
  int size_out[3];          /* Size of the output grid */
#ifdef GPAW_CUDA
  int cuda;
#endif
} TransformerObject;

#ifdef GPAW_CUDA
void transformer_init_cuda(TransformerObject *self);
void transformer_delete_cuda(TransformerObject *self);
#endif
#endif
