#include <string.h>
#include <assert.h>
#include "../bc.h"
#include "../extensions.h"
#include "gpaw-cuda.h"
#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include <cuda_runtime_api.h>



static cudaStream_t bc_send_stream[3][2];
static cudaStream_t bc_recv_stream[3][2];

static int bc_streams = 0;

static double *bc_rbuff=NULL;
static double *bc_sbuff=NULL;
static double *bc_rbuff_gpu=NULL;
static double *bc_sbuff_gpu=NULL;

static int bc_rbuff_size=0;
static int bc_sbuff_size=0;
static int bc_rbuff_max=0;
static int bc_sbuff_max=0;

void bc_init_cuda(boundary_conditions* bc)
{
  int nsends=bc->nsend[0][0]+bc->nsend[0][1]+bc->nsend[1][0]+bc->nsend[1][1]+
    bc->nsend[2][0]+bc->nsend[2][1];
  
  int nrecvs=bc->nrecv[0][0]+bc->nrecv[0][1]+bc->nrecv[1][0]+bc->nrecv[1][1]+
    bc->nrecv[2][0]+bc->nrecv[2][1];

  bc_sbuff_max=MAX(nsends, bc_sbuff_max);
  bc_rbuff_max=MAX(nrecvs, bc_rbuff_max);
  

}

void bc_init_buffers(const boundary_conditions* bc,int blocks)
{
 
  int nsends=(bc->nsend[0][0]+bc->nsend[0][1]+bc->nsend[1][0]+bc->nsend[1][1]+
    bc->nsend[2][0]+bc->nsend[2][1])*blocks;
  
  int nrecvs=(bc->nrecv[0][0]+bc->nrecv[0][1]+bc->nrecv[1][0]+bc->nrecv[1][1]+
    bc->nrecv[2][0]+bc->nrecv[2][1])*blocks;

  bc_sbuff_max=MAX(nsends, bc_sbuff_max);  
  if (bc_sbuff_max > bc_sbuff_size) {
    if (bc_sbuff) cudaFreeHost(bc_sbuff);
    GPAW_CUDAMALLOC_HOST(&bc_sbuff,double, bc_sbuff_max);
    if (bc_sbuff_gpu) cudaFree(bc_sbuff_gpu);
    GPAW_CUDAMALLOC(&bc_sbuff_gpu,double, bc_sbuff_max);
    bc_sbuff_size=bc_sbuff_max;
  }

  bc_rbuff_max=MAX(nrecvs, bc_rbuff_max);
  if (bc_rbuff_max > bc_rbuff_size) {
    if (bc_rbuff) cudaFreeHost(bc_rbuff);
    GPAW_CUDAMALLOC_HOST(&bc_rbuff,double, bc_rbuff_max);
    if (bc_rbuff_gpu) cudaFree(bc_rbuff_gpu);
    GPAW_CUDAMALLOC(&bc_rbuff_gpu,double, bc_rbuff_max);
    bc_rbuff_size=bc_rbuff_max;
  }
}

void bc_unpack_cuda_gpu_all(const boundary_conditions* bc,
			    const double* aa1, double* aa2, 
			    MPI_Request recvreq[3][2],
			    MPI_Request sendreq[3][2],
			    const double_complex *phases,
			    cudaStream_t thd, int nin)
{
  
  int ng2 = bc->ndouble * bc->size2[0] * bc->size2[1] * bc->size2[2];
  bool real = (bc->ndouble == 1);
  
  bc_init_buffers(bc,nin);
  // Copy data from a1 to central part of a2 and zero boundaries:
  if (real)
    bmgs_paste_zero_cuda_gpu(aa1, bc->size1, aa2,
			     bc->size2, bc->sendstart[0][0],nin,thd);
  else
    bmgs_paste_zero_cuda_gpuz((const cuDoubleComplex*)(aa1), 
			      bc->size1, (cuDoubleComplex*)aa2,
			      bc->size2, bc->sendstart[0][0],nin,thd);
  
#ifdef PARALLEL
  // Start receiving.
  double *rbuf=bc_rbuff;
  for (int i = 0; i < 3; i++) {
    for (int d = 0; d < 2; d++) {
      int p = bc->recvproc[i][d];
      recvreq[i][d]=0;
      if (p >= 0) {
	assert(MPI_Irecv(rbuf, bc->nrecv[i][d] * nin, MPI_DOUBLE, p,
		  d + 1000 * i,
 		  bc->comm, &recvreq[i][d])==MPI_SUCCESS);
	rbuf+=bc->nrecv[i][d] * nin;
      }
    }
  }
  
  // Prepare send-buffers
  double *sbuf_gpu=bc_sbuff_gpu;
  for (int i = 0; i < 3; i++) {
    for (int d = 0; d < 2; d++) {
      if (bc->sendproc[i][d] >= 0) {
	const int* start = bc->sendstart[i][d];
	const int* size = bc->sendsize[i][d];
	if (real)
	  bmgs_cut_cuda_gpu(aa2, bc->size2, start,
			    sbuf_gpu,
			    size,nin ,thd);
	else {
	  cuDoubleComplex phase={creal(phases[d+2*i]),cimag(phases[d*i])};
	  bmgs_cut_cuda_gpuz((cuDoubleComplex*)(aa2), bc->size2, 
			     start,
			     (cuDoubleComplex*)(sbuf_gpu),
			     size, phase, nin, thd);
	}
	sbuf_gpu += bc->nsend[i][d] * nin;
      }
    }
  }
  if  ((bc->sendproc[0][0] >= 0) || (bc->sendproc[0][1] >= 0) ||
       (bc->sendproc[1][0] >= 0) || (bc->sendproc[1][1] >= 0) ||
       (bc->sendproc[2][0] >= 0) || (bc->sendproc[2][1] >= 0))
    GPAW_CUDAMEMCPY(bc_sbuff,bc_sbuff_gpu,double,
		    (bc->nsend[0][0]+bc->nsend[0][1]+bc->nsend[1][0]+
		     bc->nsend[1][1]+bc->nsend[2][0]+bc->nsend[2][1])*nin,
		    cudaMemcpyDeviceToHost);
  
  double *sbuf=bc_sbuff;
  for (int i = 0; i < 3; i++) {
    for (int d = 0; d < 2; d++) {
      sendreq[i][d] = 0;
      int p = bc->sendproc[i][d];
      if (p >= 0) {
	//Start sending:	       	
	assert(MPI_Isend( sbuf, bc->nsend[i][d] * nin, MPI_DOUBLE, p,
			  1 - d + 1000 * i, bc->comm, 
			  &sendreq[i][d])==MPI_SUCCESS);
	sbuf += bc->nsend[i][d] * nin;	
      }
    }
  }
    // }
#endif // Parallel

    // Copy data for periodic boundary conditions:
  for (int i = 0; i < 3; i++) {          
    for (int d = 0; d < 2; d++)
      if (bc->sendproc[i][d] == COPY_DATA)  {      
	if (real) {
	  bmgs_translate_cuda_gpu(aa2, bc->size2, bc->sendsize[i][d],
				  bc->sendstart[i][d], bc->recvstart[i][1 - d],
				  nin,thd);
	} else {
	  cuDoubleComplex phase={creal(phases[d]),cimag(phases[d])};
	  bmgs_translate_cuda_gpuz((cuDoubleComplex*)(aa2), 
				   bc->size2,bc->sendsize[i][d],
				   bc->sendstart[i][d],bc->recvstart[i][1 - d],
				   phase,nin,thd);
	}
      }
  }
  
#ifdef PARALLEL
  
    
  // Store data from receive-buffer:
  
  for (int i = 0; i < 3; i++) {    
    for (int d = 0; d < 2; d++)
      if (bc->recvproc[i][d] >= 0)  {
	assert(MPI_Wait(&recvreq[i][d], MPI_STATUS_IGNORE)==MPI_SUCCESS);
      }
  }
  if  ((bc->recvproc[0][0] >= 0) || (bc->recvproc[0][1] >= 0) ||
       (bc->recvproc[1][0] >= 0) || (bc->recvproc[1][1] >= 0) ||
       (bc->recvproc[2][0] >= 0) || (bc->recvproc[2][1] >= 0))
    GPAW_CUDAMEMCPY(bc_rbuff_gpu,bc_rbuff,double,
		    (bc->nrecv[0][0]+bc->nrecv[0][1]+bc->nrecv[1][0]+
		     bc->nrecv[1][1]+bc->nrecv[2][0]+bc->nrecv[2][1])*nin,
		    cudaMemcpyHostToDevice);
  
  double *rbuf_gpu=bc_rbuff_gpu;
  for (int i = 0; i < 3; i++) {    
    for (int d = 0; d < 2; d++)
      if (bc->recvproc[i][d] >= 0)  {
	if (real)
	  bmgs_paste_cuda_gpu(rbuf_gpu, bc->recvsize[i][d],
			      aa2, bc->size2, bc->recvstart[i][d],nin,thd);
	
	else
	  bmgs_paste_cuda_gpuz((const cuDoubleComplex*)(rbuf_gpu),
			       bc->recvsize[i][d],
			       (cuDoubleComplex*)(aa2),
			       bc->size2, bc->recvstart[i][d],nin,thd);
	rbuf_gpu+=bc->nrecv[i][d]*nin;
      }
  }
  // This does not work on the ibm with gcc!  We do a blocking send instead.
  for (int i = 0; i < 3; i++) {  
    for (int d = 0; d < 2; d++)
      if (bc->sendproc[i][d] >= 0)
	assert(MPI_Wait(&sendreq[i][d], MPI_STATUS_IGNORE)==MPI_SUCCESS);
  }
#endif // PARALLEL
}


void bc_unpack_cuda_gpu(const boundary_conditions* bc,
			const double* aa1, double* aa2, int i,
			MPI_Request recvreq[2],
			MPI_Request sendreq[2],
			const double_complex phases[2], int thd, int nin)
{
  
  int ng = bc->ndouble * bc->size1[0] * bc->size1[1] * bc->size1[2];
  int ng2 = bc->ndouble * bc->size2[0] * bc->size2[1] * bc->size2[2];
  bool real = (bc->ndouble == 1);

  bc_init_buffers(bc,nin);
  if ((i == 0)) {
    // Copy data:
    // Zero all of a2 array.  We should only zero the bounaries
    // that are not periodic, but it's simpler to zero everything!
    /*    gpaw_cudaSafeCall(cuMemsetD32((CUdeviceptr)(aa2),0, 
	  nin*sizeof(double)*ng2/4));    */
    
    // Copy data from a1 to central part of a2 and zero boundaries:
    if (real)
      bmgs_paste_zero_cuda_gpu(aa1, bc->size1, aa2,
			       bc->size2, bc->sendstart[0][0],nin,0);
    else
      bmgs_paste_zero_cuda_gpuz((const cuDoubleComplex*)(aa1), 
				bc->size1, (cuDoubleComplex*)aa2,
				bc->size2, bc->sendstart[0][0],nin,0);
  }

#ifdef PARALLEL
  // Start receiving.
  double *rbuf=bc_rbuff;
  for (int d = 0; d < 2; d++) {
    int p = bc->recvproc[i][d];
    if (p >= 0) {
      assert(MPI_Irecv(rbuf, bc->nrecv[i][d] * nin, MPI_DOUBLE, p,
		       d + 10 * thd + 1000 * i,
		       bc->comm, &recvreq[d])==MPI_SUCCESS);
      rbuf += bc->nrecv[i][d] * nin;
    }
  }
  // Prepare send-buffers
  
  double* sbuf_gpu = bc_sbuff_gpu;

  for (int d = 0; d < 2; d++) {
    if (bc->sendproc[i][d] >= 0) {
      const int* start = bc->sendstart[i][d];
      const int* size = bc->sendsize[i][d];
      if (real)
	bmgs_cut_cuda_gpu(aa2, bc->size2, start,
			  sbuf_gpu,
			  size,nin,0);
      else {
	cuDoubleComplex phase={creal(phases[d]),cimag(phases[d])};
	bmgs_cut_cuda_gpuz((cuDoubleComplex*)(aa2), bc->size2, 
			   start,
			   (cuDoubleComplex*)(sbuf_gpu),
			   size, phase, nin, 0);
      }
      sbuf_gpu += bc->nsend[i][d] * nin;
    }
  }
  if (bc->sendproc[i][0]>=0 || bc->sendproc[i][1]>=0)      
    cudaMemcpy(bc_sbuff,bc_sbuff_gpu,
	       (bc->nsend[i][0]+bc->nsend[i][1]) *  nin * sizeof(double), 
	       cudaMemcpyDeviceToHost);

  //Start sending:
  double* sbuf = bc_sbuff;
  for (int d = 0; d < 2; d++) {
    sendreq[d] = 0;
    int p = bc->sendproc[i][d];
    if (p >= 0) {
      assert(MPI_Isend(sbuf, bc->nsend[i][d] * nin, MPI_DOUBLE, p,
		       1 - d + 10 * thd + 1000 * i, bc->comm, 
		       &sendreq[d])==MPI_SUCCESS);
      sbuf += bc->nsend[i][d] * nin;
      
    }
  } 
  
#endif // Parallel
  // Copy data for periodic boundary conditions:
  for (int d = 0; d < 2; d++) {
    if (bc->sendproc[i][d] == COPY_DATA)  {      
      if (real) {
	bmgs_translate_cuda_gpu(aa2 + 0 * ng2, bc->size2, bc->sendsize[i][d],
				bc->sendstart[i][d], bc->recvstart[i][1 - d],
				nin,0);
      } else {
	cuDoubleComplex phase={creal(phases[d]),cimag(phases[d])};
	bmgs_translate_cuda_gpuz((cuDoubleComplex*)(aa2 + 0 * ng2), 
				 bc->size2,bc->sendsize[i][d],
				 bc->sendstart[i][d],bc->recvstart[i][1 - d],
				 phase,nin,0);
      }
    }
  }
#ifdef PARALLEL
  for (int d = 0; d < 2; d++) {
    if (bc->recvproc[i][d] >= 0)  {
      assert(MPI_Wait(&recvreq[d],  MPI_STATUS_IGNORE)==MPI_SUCCESS);
    }
  }
  if (bc->recvproc[i][0]>=0 || bc->recvproc[i][1]>=0)      
    cudaMemcpy(bc_rbuff_gpu,bc_rbuff,
	       (bc->nrecv[i][0]+bc->nrecv[i][1]) *  nin * sizeof(double), 
	       cudaMemcpyHostToDevice);
  
  double *rbuf_gpu=bc_rbuff_gpu;
  for (int d = 0; d < 2; d++) {
    if (bc->recvproc[i][d] >= 0)  {
      if (real)
	bmgs_paste_cuda_gpu(rbuf_gpu, bc->recvsize[i][d],
			    aa2, bc->size2, bc->recvstart[i][d],nin,0);
      
      else
	bmgs_paste_cuda_gpuz((const cuDoubleComplex*)(rbuf_gpu),
			     bc->recvsize[i][d],
			     (cuDoubleComplex*)(aa2),
			     bc->size2, bc->recvstart[i][d],nin,0);      
      rbuf_gpu += bc->nrecv[i][d] * nin;
    }
  }
  // This does not work on the ibm with gcc!  We do a blocking send instead.
  for (int d = 0; d < 2; d++)
    if (bc->sendproc[i][d] >= 0)
      assert(MPI_Wait(&sendreq[d],  MPI_STATUS_IGNORE)==MPI_SUCCESS);
#endif // Parallel  
  
}


void bc_unpack1_cuda_gpu_async_all(const boundary_conditions* bc,
				   const double* aa1, double* aa2,
				   MPI_Request recvreq[3][2],
				   MPI_Request sendreq[3][2],
				   const double_complex *phases,
				   cudaStream_t thd, int nin)
{
  bool real = (bc->ndouble == 1);
  
  bc_init_buffers(bc,nin);  
  // Copy data from a1 to central part of a2 and zero boundaries:
  if (real)
    bmgs_paste_zero_cuda_gpu(aa1, bc->size1, aa2,
			     bc->size2, bc->sendstart[0][0],nin,thd);
  else
    bmgs_paste_zero_cuda_gpuz((const cuDoubleComplex*)(aa1), 
			      bc->size1, (cuDoubleComplex*)aa2,
			      bc->size2, bc->sendstart[0][0],nin,thd);

  double* rbuf = bc_rbuff;
    
  // Start receiving.
  for (int i = 0; i < 3; i++) {
    for (int d = 0; d < 2; d++) {
      int p = bc->recvproc[i][d];
      if (p >= 0) {

	MPI_Irecv(rbuf, bc->nrecv[i][d]*nin,  MPI_DOUBLE, p,
		  d + 1000 * i,  bc->comm, &recvreq[i][d]);
	rbuf+=bc->nrecv[i][d]*nin;
      }
    }
  }

  double* sbuf_gpu = bc_sbuff_gpu;
  for (int i = 0; i < 3; i++) {    
    for (int d = 0; d < 2; d++) {
      int p = bc->sendproc[i][d];
      if (p >= 0) {
	const int* start = bc->sendstart[i][d];
	const int* size = bc->sendsize[i][d];
	if (real)
	  bmgs_cut_cuda_gpu(aa2, bc->size2, start,
			    sbuf_gpu,
			    size,nin ,thd);
	else {
	  cuDoubleComplex phase={creal(phases[d+2*i]),cimag(phases[d+2*i])};
	  bmgs_cut_cuda_gpuz((cuDoubleComplex*)(aa2), bc->size2, 
			     start,
			     (cuDoubleComplex*)(sbuf_gpu),
			     size, phase, nin, thd);
	}
	sbuf_gpu+=bc->nsend[i][d]*nin;
      }
    } 
  }  
  cudaStreamSynchronize(thd);    
  
}


void bc_unpack2_cuda_gpu_async_all(const boundary_conditions* bc,
				   const double* aa1, double* aa2,
				   MPI_Request recvreq[3][2],
				   MPI_Request sendreq[3][2],
				   const double_complex *phases,
				   cudaStream_t thd, int nin)
{
  
  int ng = bc->ndouble * bc->size1[0] * bc->size1[1] * bc->size1[2];
  int ng2 = bc->ndouble * bc->size2[0] * bc->size2[1] * bc->size2[2];
  bool real = (bc->ndouble == 1);
  
  
#ifdef PARALLEL
  
  if  (!bc_streams){
    for (int i1=0;i1<3;i1++)
      for (int i2=0;i2<2;i2++){
	  cudaStreamCreate(&(bc_send_stream[i1][i2]));
	  cudaStreamCreate(&(bc_recv_stream[i1][i2]));
	  
      }
    bc_streams=2*3;
  }

  int sendp=0,recvp=0;
  double *sbuff[3][2];
  double *rbuff[3][2];
  double *sbuff_gpu[3][2];
  double *rbuff_gpu[3][2];
  
  for (int i=0;i<3;i++){
    for (int d=0;d<2;d++){
      sbuff[i][d]=bc_sbuff+sendp;
      sbuff_gpu[i][d]=bc_sbuff_gpu+sendp;
      if (bc->sendproc[i][d]>=0) {
	sendp+=bc->nsend[i][d]*nin;
      }
      rbuff[i][d]=bc_rbuff+recvp;
      rbuff_gpu[i][d]=bc_rbuff_gpu+recvp;
      if (bc->recvproc[i][d]>=0) {
	recvp+=bc->nrecv[i][d]*nin;
      }
    }
  }

  double* rbuf_gpu;  
  double* rbuf;  
  double* sbuf_gpu;
  double* sbuf;
  int all_done;
  int recv_done[3][2]={0,0,0,0,0,0};
  
  
  // Prepare send-buffers

  for (int i = 0; i < 3; i++) {
    for (int d = 0; d < 2; d++)  {
      sendreq[i][d] = 0;
    }
  }
  for (int i = 0; i < 3; i++) {
    for (int d = 0; d < 2; d++) {
      if (bc->sendproc[i][d] >= 0) {
	GPAW_CUDAMEMCPY_A(sbuff[i][d], sbuff_gpu[i][d],double,
			  bc->nsend[i][d]*nin,
			  cudaMemcpyDeviceToHost,bc_send_stream[i][d]);
      }
    }
  } 
  
  for (int i = 0; i < 3; i++) {
    for (int d = 0; d < 2; d++)  {
      if ((bc->sendproc[i][d]>= 0)) {
	do{
	  for (int ii = 0; ii < 3; ii++) {
	    for (int dd = 0; dd < 2; dd++)  {
	      if ((bc->recvproc[ii][dd] >= 0) && 
		  (!recv_done[ii][dd])) {
		int status;
		MPI_Test(&recvreq[ii][dd],&status, MPI_STATUS_IGNORE);
		if (status){
		  GPAW_CUDAMEMCPY_A(rbuff_gpu[ii][dd], rbuff[ii][dd],double,
				    bc->nrecv[ii][dd]*nin,
				    cudaMemcpyHostToDevice,
				    bc_recv_stream[ii][dd]);
		  recv_done[ii][dd]=1; 
		}		      
	      }
	    }
	  }	  
	} while (cudaStreamQuery(bc_send_stream[i][d])!=cudaSuccess);
	MPI_Isend(sbuff[i][d], 
		  bc->nsend[i][d]*nin , MPI_DOUBLE, 
		  bc->sendproc[i][d],
		  1 - d + 1000 * i, bc->comm, 
		  &sendreq[i][d]);
	
      }
    }
  }
  
  do {
    all_done=1;
    for (int ii = 0; ii < 3; ii++) {
      for (int dd = 0; dd < 2; dd++)  {
	if ((bc->recvproc[ii][dd] >= 0) && 
	    (!recv_done[ii][dd])) {
	  int status;
	  MPI_Test(&recvreq[ii][dd],&status, MPI_STATUS_IGNORE);
	  if (status){
	    GPAW_CUDAMEMCPY_A(rbuff_gpu[ii][dd], rbuff[ii][dd], double,
			      bc->nrecv[ii][dd]*nin,
			      cudaMemcpyHostToDevice,
			      bc_recv_stream[ii][dd]);
	    recv_done[ii][dd]=1; 
	  }		      
	  all_done&=recv_done[ii][dd];
	}
      }
    }
  }  while (!all_done);
  
  // This does not work on the ibm with gcc!  We do a blocking send instead.
  
#endif // Parallel
  
  // Copy data for periodic boundary conditions:
  for (int i = 0; i < 3; i++) {
    for (int d = 0; d < 2; d++)
      if (bc->sendproc[i][d] == COPY_DATA)  {      
	if (real) {
	  bmgs_translate_cuda_gpu(aa2, bc->size2, bc->sendsize[i][d],
				  bc->sendstart[i][d], bc->recvstart[i][1 - d],
				  nin,thd);
	} else {
	  cuDoubleComplex phase={creal(phases[d+2*i]),cimag(phases[d+2*i])};
	  bmgs_translate_cuda_gpuz((cuDoubleComplex*)(aa2), 
				   bc->size2,bc->sendsize[i][d],
				   bc->sendstart[i][d],bc->recvstart[i][1 - d],
				   phase,nin,thd);
	}
      }
  }
#ifdef PARALLEL

  for (int ii = 0; ii < 3; ii++) {
    for (int dd = 0; dd < 2; dd++)  {
      if (bc->recvproc[ii][dd] >= 0)  {
	cudaStreamSynchronize(bc_recv_stream[ii][dd]);
	if (real)
	  bmgs_paste_cuda_gpu(rbuff_gpu[ii][dd], bc->recvsize[ii][dd],
			      aa2, bc->size2, bc->recvstart[ii][dd],nin,
			      thd);      
	else
	  bmgs_paste_cuda_gpuz((const cuDoubleComplex*)(rbuff_gpu[ii][dd]),
			       bc->recvsize[ii][dd],
			       (cuDoubleComplex*)(aa2),
			       bc->size2, bc->recvstart[ii][dd],nin,
			       thd);		
      }
    }
  }
  
  for (int i = 0; i < 3; i++) 
    for (int d = 0; d < 2; d++)
	if (sendreq[i][d] != 0)
	  MPI_Wait(&sendreq[i][d], MPI_STATUS_IGNORE);
  
#endif // Parallel
    
}


