#include <string.h>
#include <assert.h>
#include "../bc.h"
#include "../extensions.h"
#include "gpaw-cuda.h"
#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include <cuda_runtime_api.h>


void bc_unpack1_cuda_gpu(const boundary_conditions* bc,
			 double* aa1, double* aa2, int i,
			 MPI_Request recvreq[2],
			 MPI_Request sendreq[2],
			 double* rbuff, double* sbuff,
			 const double_complex phases[2], int thd, int nin)
{

  int ng = bc->ndouble * bc->size1[0] * bc->size1[1] * bc->size1[2];
  int ng2 = bc->ndouble * bc->size2[0] * bc->size2[1] * bc->size2[2];
  bool real = (bc->ndouble == 1);

  for (int m = 0; m < nin; m++)
    // Copy data:
    if (i == 0)
      {
	//	fprintf(stdout,"unpack1 copy data\n");
        // Zero all of a2 array.  We should only zero the bounaries
        // that are not periodic, but it's simpler to zero everything!
        // XXX
	//        memset(aa2 + m * ng2, 0, ng2 * sizeof(double));
	gpaw_cudaSafeCall(cuMemsetD32((CUdeviceptr)(aa2 + m * ng2),0, sizeof(double)*ng2/4));
	/*	gpaw_cudaSafeCall(cudaGetLastError());
	gpaw_cudaSafeCall(cudaMemset((aa2 + m * ng2),0, sizeof(double)*ng2));
	gpaw_cudaSafeCall(cudaMemset((aa1 + m * ng),0, sizeof(double)*ng));
	gpaw_cudaSafeCall(cudaGetLastError());
	printf("m %d\n",m);
	printf("ng %d\n",ng);
	printf("ng2 %d\n",ng2);
	printf("size1 %d %d %d\n",bc->size1[0],bc->size1[1],bc->size1[2]);
	printf("size2 %d %d %d\n",bc->size2[0],bc->size2[1],bc->size2[2]);*/
        // Copy data from a1 to central part of a2:
	if (real)
	  bmgs_paste_cuda_gpu(aa1 + m * ng, bc->size1, aa2 + m * ng2,
			      bc->size2, bc->sendstart[0][0]);
        else
	  bmgs_paste_cuda_gpuz((const cuDoubleComplex*)(aa1 + m * ng), 
			       bc->size1, (cuDoubleComplex*)aa2 + m * ng2,
			       bc->size2, bc->sendstart[0][0]);
      }

#ifdef PARALLEL

  // Start receiving.
  for (int d = 0; d < 2; d++)
    {
      int p = bc->recvproc[i][d];
      if (p >= 0)
        {
	  //	  fprintf(stdout,"unpack1 Irecv\n");
          if (bc->rjoin[i])
            {
              if (d == 0)
                MPI_Irecv(rbuff, (bc->nrecv[i][0] + bc->nrecv[i][1]) * nin,
			  MPI_DOUBLE, p,
                          10 * thd + 1000 * i + 100000,
                          bc->comm, &recvreq[0]);
            }
          else
          {
            MPI_Irecv(rbuff, bc->nrecv[i][d] * nin, MPI_DOUBLE, p,
		      d + 10 * thd + 1000 * i,
                      bc->comm, &recvreq[d]);
	    rbuff += bc->nrecv[i][d] * nin;
          }
        }
    }
  // Prepare send-buffers and start sending:
  double* sbuf = sbuff;
  double* sbuf0 = sbuff;
  for (int d = 0; d < 2; d++)
    {
      sendreq[d] = 0;
      int p = bc->sendproc[i][d];
      if (p >= 0)
        {
          const int* start = bc->sendstart[i][d];
          const int* size = bc->sendsize[i][d];

	  for (int m = 0; m < nin; m++)
	    if (real)
	      bmgs_cut_cuda(aa2 + m * ng2, bc->size2, start,
			    sbuf + m * bc->nsend[i][d],
			    size,cudaMemcpyDeviceToHost);
	    else
	      assert(0);

          if (bc->sjoin[i])
            {
              if (d == 1)
                {
                  MPI_Isend(sbuf0, (bc->nsend[i][0] + bc->nsend[i][1]) * nin,
			    MPI_DOUBLE, p,
                            10 * thd + 1000 * i + 100000,
                            bc->comm, &sendreq[0]);
                }
            }
          else
            {
              MPI_Isend(sbuf, bc->nsend[i][d] * nin, MPI_DOUBLE, p,
                        1 - d + 10 * thd + 1000 * i, bc->comm, &sendreq[d]);
            }
          sbuf += bc->nsend[i][d] * nin;
        }
    }
#endif // Parallel
  for (int m = 0; m < nin; m++)
    {
      // Copy data for periodic boundary conditions:
      for (int d = 0; d < 2; d++)
        if (bc->sendproc[i][d] == COPY_DATA)
          {
	    // fprintf(stdout,"unpack1 copy data periodic\n");
            if (real)
              bmgs_translate_cuda(aa2 + m * ng2, bc->size2, bc->sendsize[i][d],
				  bc->sendstart[i][d], bc->recvstart[i][1 - d],
				  cudaMemcpyDeviceToDevice);
            else {
	      cuDoubleComplex phase={creal(phases[d]),cimag(phases[d])};
              bmgs_translate_cudaz((cuDoubleComplex*)(aa2 + m * ng2), 
				   bc->size2,bc->sendsize[i][d],
				   bc->sendstart[i][d],bc->recvstart[i][1 - d],
				   phase,cudaMemcpyDeviceToDevice);
	    }
          }
    }
}


void bc_unpack2_cuda_gpu(const boundary_conditions* bc,
    double* a2, int i,
    MPI_Request recvreq[2],
    MPI_Request sendreq[2],
    double* rbuf, int nin)
{
#ifdef PARALLEL


  int ng2 = bc->ndouble * bc->size2[0] * bc->size2[1] * bc->size2[2];

  // Store data from receive-buffer:
  bool real = (bc->ndouble == 1);

  double* rbuf0 = rbuf;


  for (int d = 0; d < 2; d++)
    if (bc->recvproc[i][d] >= 0)
      {
        if (bc->rjoin[i])
          {
            if (d == 0)
              {
                MPI_Wait(&recvreq[0], MPI_STATUS_IGNORE);
                rbuf += bc->nrecv[i][1] * nin;
              }
            else
              rbuf = rbuf0;
	  }
	else
	  MPI_Wait(&recvreq[d], MPI_STATUS_IGNORE);
	
	for (int m = 0; m < nin; m++)
	  if (real)
	  bmgs_paste_cuda(rbuf + m * bc->nrecv[i][d], bc->recvsize[i][d],
			  a2 + m * ng2, bc->size2, bc->recvstart[i][d],cudaMemcpyHostToDevice);

	  else
	    assert(0);

	rbuf += bc->nrecv[i][d] * nin;
      }
  
  // This does not work on the ibm with gcc!  We do a blocking send instead.
  for (int d = 0; d < 2; d++)
    if (sendreq[d] != 0)
      MPI_Wait(&sendreq[d], MPI_STATUS_IGNORE);
#endif // PARALLEL
}
