/*  Copyright (C) 2003-2007  CAMP
 *  Copyright (C) 2005       CSC - IT Center for Science Ltd.
 *  Please see the accompanying LICENSE file for further information. */

#ifndef BC_H
#define BC_H

#include "bmgs/bmgs.h"
#ifdef PARALLEL
#include <mpi.h>
#else
typedef int* MPI_Request; // !!!!!!!???????????
typedef int* MPI_Comm;
#define MPI_COMM_NULL 0
#define MPI_Comm_rank(comm, rank) *(rank) = 0
#endif


typedef struct
{
  int size1[3];
  int size2[3];
  int sendstart[3][2][3];
  int sendsize[3][2][3];
  int recvstart[3][2][3];
  int recvsize[3][2][3];
  int sendproc[3][2];
  int recvproc[3][2];
  int nsend[3][2];
  int nrecv[3][2];
  int maxsend;
  int maxrecv;
  int padding[3];
  bool sjoin[3];
  bool rjoin[3];
  int ndouble;
  bool cfd;
  MPI_Comm comm;
#ifdef GPAW_GPU
  bool gpu_sjoin[3];
  bool gpu_rjoin[3];
  bool gpu_async[3];
#endif
} boundary_conditions;

static const int COPY_DATA = -2;
static const int DO_NOTHING = -3; // ??????????

boundary_conditions* bc_init(const long size1[3],
           const long padding[3][2],
           const long npadding[3][2],
           const long neighbors[3][2],
           MPI_Comm comm, bool real, bool cfd);
void bc_unpack1(const boundary_conditions* bc,
    const double* input, double* output, int i,
    MPI_Request recvreq[2],
    MPI_Request sendreq[2],
    double* rbuf, double* sbuf,
    const double_complex phases[2], int thd, int nin);
void bc_unpack2(const boundary_conditions* bc,
    double* a2, int i,
    MPI_Request recvreq[2],
    MPI_Request sendreq[2],
    double* rbuf, int nin);

#ifdef GPAW_GPU
#include "gpu/gpu-runtime.h"

void bc_init_gpu(boundary_conditions* bc);
void bc_dealloc_gpu(int force);
void bc_unpack_gpu(const boundary_conditions* bc,
                   double* aa2, int i,
                   MPI_Request recvreq[3][2],
                   MPI_Request sendreq[2],
                   const double_complex phases[2],
                   gpuStream_t kernel_stream,
                   int nin);
void bc_unpack_gpu_async(const boundary_conditions* bc,
                         double* aa2, int i,
                         MPI_Request recvreq[3][2],
                         MPI_Request sendreq[2],
                         const double_complex phases[2],
                         gpuStream_t kernel_stream,
                         int nin);
void bc_unpack_paste_gpu(boundary_conditions* bc,
                         const double* aa1, double* aa2,
                         MPI_Request recvreq[3][2],
                         gpuStream_t thd, int nin);
#endif

#endif
