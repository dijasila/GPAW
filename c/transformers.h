#ifndef __TRANSFORMERS_H
#define __TRANSFORMERS_H

/*  Copyright (C) 2009-2012  CSC - IT Center for Science Ltd.
 *  Please see the accompanying LICENSE file for further information. */


#ifdef __TRANSFORMERS_C
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
} TransformerObject;
#else
// Provide an opaque type for routines outside transformers.c 
struct _TransformerObject;
typedef struct _TransformerObject TransformerObject;

#endif

void transapply_worker(TransformerObject *self, int chunksize, int start,
		  int end, int thread_id, int nthreads,
		  const double* in, double* out,
		  bool real, const double_complex* ph);
#endif
