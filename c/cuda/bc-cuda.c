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
			 const double* aa1, double* aa2, int i,
			 MPI_Request recvreq[2],
			 MPI_Request sendreq[2],
			 double* rbuff, double* sbuff,  
			 double* sbuff_gpu,
			 const double_complex phases[2], int thd, int nin)
{


  int ng = bc->ndouble * bc->size1[0] * bc->size1[1] * bc->size1[2];
  int ng2 = bc->ndouble * bc->size2[0] * bc->size2[1] * bc->size2[2];
  bool real = (bc->ndouble == 1);

  if ((i == 0)) {
    // Copy data:
    // Zero all of a2 array.  We should only zero the bounaries
    // that are not periodic, but it's simpler to zero everything!
    /*    gpaw_cudaSafeCall(cuMemsetD32((CUdeviceptr)(aa2),0, 
	  nin*sizeof(double)*ng2/4));    */
    
    // Copy data from a1 to central part of a2 and zero boundaries:
    if (real)
      bmgs_paste_zero_cuda_gpu(aa1, bc->size1, aa2,
			  bc->size2, bc->sendstart[0][0],nin);
    else
      bmgs_paste_zero_cuda_gpuz((const cuDoubleComplex*)(aa1), 
				bc->size1, (cuDoubleComplex*)aa2,
				bc->size2, bc->sendstart[0][0],nin);
    
  }

#ifdef PARALLEL
  // Start receiving.
  
  for (int d = 0; d < 2; d++)
    {
      int p = bc->recvproc[i][d];
      if (p >= 0)
	{
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
  // Prepare send-buffers

  double* sbuf_gpu = sbuff_gpu;
  for (int d = 0; d < 2; d++)
    {
      int p = bc->sendproc[i][d];
      if (p >= 0)
        {
          const int* start = bc->sendstart[i][d];
          const int* size = bc->sendsize[i][d];
	  //	  for (int m = 0; m < nin; m++)
	  if (real)
	    bmgs_cut_cuda_gpu(aa2, bc->size2, start,
			      sbuf_gpu,
			      size,nin);
	  else {
	    cuDoubleComplex phase={creal(phases[d]),cimag(phases[d])};
	    bmgs_cut_cuda_gpuz((cuDoubleComplex*)(aa2), bc->size2, 
			       start,
			       (cuDoubleComplex*)(sbuf_gpu),
			       size, phase, nin);
	  }
	  sbuf_gpu += bc->nsend[i][d] * nin;
        }
    } 
  double* sbuf_gpu0 = sbuff_gpu; 
  double* sbuf0 = sbuff;
  if (bc->sendproc[i][0]>=0 || bc->sendproc[i][1]>=0)   
    cudaMemcpy(sbuf0,sbuf_gpu0,
	       (bc->nsend[i][0] + bc->nsend[i][1]) *  nin * sizeof(double), 
	       cudaMemcpyDeviceToHost);
  
#endif // Parallel

#ifdef PARALLEL
  //Start sending:
  double* sbuf = sbuff;

  for (int d = 0; d < 2; d++)
    {
      sendreq[d] = 0;
      int p = bc->sendproc[i][d];
      if (p >= 0)
        {
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
  // Copy data for periodic boundary conditions:
  for (int d = 0; d < 2; d++)
    if (bc->sendproc[i][d] == COPY_DATA)  {      
      if (real) {
	bmgs_translate_cuda_gpu(aa2 + 0 * ng2, bc->size2, bc->sendsize[i][d],
				bc->sendstart[i][d], bc->recvstart[i][1 - d],
				nin);
      } else {
	cuDoubleComplex phase={creal(phases[d]),cimag(phases[d])};
	bmgs_translate_cuda_gpuz((cuDoubleComplex*)(aa2 + 0 * ng2), 
				 bc->size2,bc->sendsize[i][d],
				 bc->sendstart[i][d],bc->recvstart[i][1 - d],
				 phase,nin);
      }
    }
}


void bc_unpack2_cuda_gpu(const boundary_conditions* bc,
			 double* a2, int i,
			 MPI_Request recvreq[2],
			 MPI_Request sendreq[2],
			 double* rbuf, double *rbuf_gpu,int nin)
{
#ifdef PARALLEL


  int ng2 = bc->ndouble * bc->size2[0] * bc->size2[1] * bc->size2[2];

  // Store data from receive-buffer:
  bool real = (bc->ndouble == 1);

  double* rbuf0 = rbuf;
  double* rbuf_gpu0 = rbuf_gpu;


  for (int d = 0; d < 2; d++)
    if (bc->recvproc[i][d] >= 0)
      {
        if (bc->rjoin[i])
          {
            if (d == 0)
              {
                MPI_Wait(&recvreq[0], MPI_STATUS_IGNORE);
                rbuf += bc->nrecv[i][1] * nin;
                rbuf_gpu += bc->nrecv[i][1] * nin;
              }
            else 
	      {
		rbuf = rbuf0;
		rbuf_gpu = rbuf_gpu0;
	      }
	  }
	else
	  MPI_Wait(&recvreq[d], MPI_STATUS_IGNORE);
	
	cudaMemcpy(rbuf_gpu,rbuf,
		   bc->nrecv[i][d] * nin * sizeof(double),
		   cudaMemcpyHostToDevice);		
	//	for (int m = 0; m < nin; m++)
	if (real)
	  bmgs_paste_cuda_gpu(rbuf_gpu, bc->recvsize[i][d],
			      a2, bc->size2, bc->recvstart[i][d],nin);
	
	else
	  bmgs_paste_cuda_gpuz((const cuDoubleComplex*)(rbuf_gpu),
			       bc->recvsize[i][d],
			       (cuDoubleComplex*)(a2),
			       bc->size2, bc->recvstart[i][d],nin);
	
	rbuf += bc->nrecv[i][d] * nin;
	rbuf_gpu += bc->nrecv[i][d] * nin;
      }
  
  // This does not work on the ibm with gcc!  We do a blocking send instead.
  for (int d = 0; d < 2; d++)
    if (sendreq[d] != 0)
      MPI_Wait(&sendreq[d], MPI_STATUS_IGNORE);
#endif // PARALLEL
}
