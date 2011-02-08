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
  int alloc_blocks;
  double* buf_gpu;
  double* buf2_gpu;
  double *sendbuf;
  double *recvbuf;
  double *sendbuf_gpu;
  double *recvbuf_gpu;
#endif
} TransformerObject;

#endif
