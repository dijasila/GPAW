#ifndef TRANSFORMERS_H
#define TRANSFORMERS_H

#include "bc.h"

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
  double* buf;
  double* buf2;
  double* sendbuf;
  double* recvbuf;
#ifdef GPAW_CUDA
  int cuda;
  double* buf_gpu;
  double* buf2_gpu;
#endif
} TransformerObject;

#endif
