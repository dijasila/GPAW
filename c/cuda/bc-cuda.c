#include <string.h>
#include <assert.h>
#include "../bc.h"
#include "../extensions.h"
#include "gpaw-cuda.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <cuda.h>
#include <cuda_runtime_api.h>



static cudaStream_t bc_recv_stream;

static int bc_streams = 0;

static cudaEvent_t bc_sendcpy_event[3];
static cudaEvent_t bc_recv_event[3][2];
static int bc_recv_done[3][2];
static int bc_join[3];
static int bc_async[3];

static int bc_init_count = 0;

static double *bc_rbuffs=NULL;
static double *bc_sbuffs=NULL;
static double *bc_rbuffs_gpu=NULL;
static double *bc_sbuffs_gpu=NULL;
static double *bc_rbuff[3][2];
static double *bc_sbuff[3][2];
static double *bc_rbuff_gpu[3][2];
static double *bc_sbuff_gpu[3][2];

static int bc_rbuffs_size=0;
static int bc_sbuffs_size=0;
static int bc_rbuffs_max=0;
static int bc_sbuffs_max=0;

void bc_init_cuda(boundary_conditions* bc)
{
  int nsends=(bc->nsend[0][0]+bc->nsend[0][1]+
	      bc->nsend[1][0]+bc->nsend[1][1]+
	      bc->nsend[2][0]+bc->nsend[2][1]);

  int nrecvs=(bc->nrecv[0][0]+bc->nrecv[0][1]+
	      bc->nrecv[1][0]+bc->nrecv[1][1]+
	      bc->nrecv[2][0]+bc->nrecv[2][1]);

  bc_sbuffs_max=MAX(nsends, bc_sbuffs_max);
  bc_rbuffs_max=MAX(nrecvs, bc_rbuffs_max);
  
  bc_init_count++;
}

void bc_init_buffers_cuda()
{
    bc_rbuffs=NULL;
    bc_sbuffs=NULL;
    bc_rbuffs_gpu=NULL;
    bc_sbuffs_gpu=NULL;    
    bc_rbuffs_size=0;
    bc_sbuffs_size=0;
    //bc_rbuffs_max=0;
    //bc_sbuffs_max=0;
    bc_init_count=0;
    bc_streams=0;
}

void bc_alloc_buffers(const boundary_conditions* bc,int blocks)
{
  int nsends=(bc->nsend[0][0]+bc->nsend[0][1]+
	      bc->nsend[1][0]+bc->nsend[1][1]+
	      bc->nsend[2][0]+bc->nsend[2][1])*blocks;

  int nrecvs=(bc->nrecv[0][0]+bc->nrecv[0][1]+
	      bc->nrecv[1][0]+bc->nrecv[1][1]+
	      bc->nrecv[2][0]+bc->nrecv[2][1])*blocks;
  

  bc_sbuffs_max=MAX(nsends, bc_sbuffs_max);  
  if (bc_sbuffs_max > bc_sbuffs_size) {
    cudaFreeHost(bc_sbuffs);
    cudaFree(bc_sbuffs_gpu);
    cudaGetLastError();
    GPAW_CUDAMALLOC_HOST(&bc_sbuffs,double, bc_sbuffs_max);
    GPAW_CUDAMALLOC(&bc_sbuffs_gpu,double, bc_sbuffs_max);
    bc_sbuffs_size=bc_sbuffs_max;
  }

  bc_rbuffs_max=MAX(nrecvs, bc_rbuffs_max);
  if (bc_rbuffs_max > bc_rbuffs_size) {
    cudaFreeHost(bc_rbuffs);
    cudaFree(bc_rbuffs_gpu);
    cudaGetLastError();
    GPAW_CUDAMALLOC_HOST(&bc_rbuffs,double, bc_rbuffs_max);
    GPAW_CUDAMALLOC(&bc_rbuffs_gpu,double, bc_rbuffs_max);
    bc_rbuffs_size=bc_rbuffs_max;
  }

  if  (!bc_streams){    
    cudaStreamCreate(&bc_recv_stream);
    bc_streams=2;
    for (int d=0;d<3;d++){
	cudaEventCreateWithFlags(&bc_sendcpy_event[d],
				 cudaEventDefault|cudaEventDisableTiming);
      for (int i=0;i<2;i++){
	cudaEventCreateWithFlags(&bc_recv_event[d][i],
				 cudaEventDefault|cudaEventDisableTiming);
      }	
    }
  }  
}

void bc_dealloc_cuda(int force)
{
  if (force || (bc_init_count==1)) {
    cudaFreeHost(bc_sbuffs);
    cudaFreeHost(bc_rbuffs);
    cudaFree(bc_sbuffs_gpu);
    cudaFree(bc_rbuffs_gpu);

    if  (bc_streams){      
      cudaStreamDestroy(bc_recv_stream);
      bc_streams=2;
      for (int d=0;d<3;d++){
	  cudaEventDestroy(bc_sendcpy_event[d]);  
	for (int i=0;i<2;i++){
	  cudaEventDestroy(bc_recv_event[d][i]);
	}
      }
    }  

    cudaGetLastError();
    bc_init_buffers_cuda();
    return;
  }
  if (bc_init_count>0) bc_init_count--;
 
}

void bc_unpack_cuda_gpu_sync(const boundary_conditions* bc,
			const double* aa1, double* aa2, int i,
			MPI_Request recvreq[3][2],
			MPI_Request sendreq[2],
			const double_complex phases[2], 
			cudaStream_t kernel_stream, int nin)
{
  
  bool real = (bc->ndouble == 1);
  bc_alloc_buffers(bc,nin);


#ifdef PARALLEL

  for (int d = 0; d < 2; d++) {
    if (bc->sendproc[i][d] >= 0) {
      const int* start = bc->sendstart[i][d];
      const int* size = bc->sendsize[i][d];
      if (real)
	bmgs_cut_cuda_gpu(aa2, bc->size2, start,
			  bc_sbuff_gpu[i][d],
			  size,nin,kernel_stream);
      else {
	cuDoubleComplex phase={creal(phases[d]),cimag(phases[d])};
	bmgs_cut_cuda_gpuz((cuDoubleComplex*)(aa2), bc->size2, 
			   start,
			   (cuDoubleComplex*)(bc_sbuff_gpu[i][d]),
			   size, phase, nin, kernel_stream);
      }
    }
  }
  if (bc->sendproc[i][0]>=0 || bc->sendproc[i][1]>=0)      
    /*    GPAW_CUDAMEMCPY_A(bc_sbuff[i][0],bc_sbuff_gpu[i][0],double,
		      (bc->nsend[i][0]+bc->nsend[i][1]) *  nin, 
		      cudaMemcpyDeviceToHost,kernel_stream);*/
    GPAW_CUDAMEMCPY(bc_sbuff[i][0],bc_sbuff_gpu[i][0],double,
		      (bc->nsend[i][0]+bc->nsend[i][1]) *  nin, 
		      cudaMemcpyDeviceToHost);

  //Start sending:
  for (int d = 0; d < 2; d++) {
    sendreq[d] = 0;
    int p = bc->sendproc[i][d];
    if (p >= 0) {
      //      cudaStreamSynchronize(kernel_stream); 
      assert(MPI_Isend(bc_sbuff[i][d], bc->nsend[i][d] * nin, MPI_DOUBLE, p,
		       1 - d + 1000 * i, bc->comm, 
		       &sendreq[d])==MPI_SUCCESS);
    }
  } 
  
#endif // Parallel
  // Copy data for periodic boundary conditions:
  for (int d = 0; d < 2; d++) {
    if (bc->sendproc[i][d] == COPY_DATA)  {      
      if (real) {
	bmgs_translate_cuda_gpu(aa2, bc->size2, bc->sendsize[i][d],
				bc->sendstart[i][d], bc->recvstart[i][1 - d],
				nin,kernel_stream);
      } else {
	cuDoubleComplex phase={creal(phases[d]),cimag(phases[d])};
	bmgs_translate_cuda_gpuz((cuDoubleComplex*)(aa2), 
				 bc->size2,bc->sendsize[i][d],
				 bc->sendstart[i][d],bc->recvstart[i][1 - d],
				 phase,nin,kernel_stream);
      }
    }
  }
#ifdef PARALLEL
  for (int d = 0; d < 2; d++) {
    if (!bc_recv_done[i][d]) {
      assert(MPI_Wait(&recvreq[i][d],  MPI_STATUS_IGNORE)==MPI_SUCCESS);
    }
  }
  if (!bc_recv_done[i][0] || !bc_recv_done[i][1]) {
    /*    GPAW_CUDAMEMCPY_A(bc_rbuff_gpu[i][0],bc_rbuff[i][0],double,
		      (bc->nrecv[i][0]+bc->nrecv[i][1]) * nin, 
		      cudaMemcpyHostToDevice,kernel_stream);*/
    GPAW_CUDAMEMCPY(bc_rbuff_gpu[i][0],bc_rbuff[i][0],double,
		      (bc->nrecv[i][0]+bc->nrecv[i][1]) * nin, 
		      cudaMemcpyHostToDevice);
    bc_recv_done[i][0]=1; 
    bc_recv_done[i][1]=1; 
  }  
  for (int d = 0; d < 2; d++) {
    if (bc->recvproc[i][d] >= 0)  {
      if (real)
	bmgs_paste_cuda_gpu(bc_rbuff_gpu[i][d], bc->recvsize[i][d],
			    aa2, bc->size2, bc->recvstart[i][d],nin,
			    kernel_stream);
      
      else
	bmgs_paste_cuda_gpuz((const cuDoubleComplex*)(bc_rbuff_gpu[i][d]),
			     bc->recvsize[i][d],
			     (cuDoubleComplex*)(aa2),
			     bc->size2, bc->recvstart[i][d],nin,
			     kernel_stream);    
    }
  }
  // This does not work on the ibm with gcc!  We do a blocking send instead.
  for (int d = 0; d < 2; d++)
    if (bc->sendproc[i][d] >= 0)
      assert(MPI_Wait(&sendreq[d],  MPI_STATUS_IGNORE)==MPI_SUCCESS);
#endif // Parallel  
}

void bc_unpack_paste_cuda_gpu(const boundary_conditions* bc,
			      const double* aa1, double* aa2,
			      MPI_Request recvreq[3][2],
			      cudaStream_t kernel_stream, int nin)

{
  bool real = (bc->ndouble == 1);

  bc_alloc_buffers(bc,nin);
  // Copy data:
  // Zero all of a2 array.  We should only zero the bounaries
  // that are not periodic, but it's simpler to zero everything!
  
  // Copy data from a1 to central part of a2 and zero boundaries:
  if (real)
    bmgs_paste_zero_cuda_gpu(aa1, bc->size1, aa2,
			     bc->size2, bc->sendstart[0][0],nin,kernel_stream);
  else
    bmgs_paste_zero_cuda_gpuz((const cuDoubleComplex*)(aa1), 
			      bc->size1, (cuDoubleComplex*)aa2,
			      bc->size2, bc->sendstart[0][0],nin,
			      kernel_stream);

  int recvp=0,sendp=0;

  for (int i=0;i<3;i++){
    for (int d=0;d<2;d++){
      bc_sbuff[i][d]=bc_sbuffs+sendp;
      bc_sbuff_gpu[i][d]=bc_sbuffs_gpu+sendp;
      if (bc->sendproc[i][d]>=0) {
	sendp+=bc->nsend[i][d]*nin;
      }
      bc_rbuff[i][d]=bc_rbuffs+recvp;
      bc_rbuff_gpu[i][d]=bc_rbuffs_gpu+recvp;
      if (bc->recvproc[i][d]>=0) {
	recvp+=bc->nrecv[i][d]*nin;
      }
    }
  }

  for (int i = 0; i < 3; i++) {
    for (int d = 0; d < 2; d++) {
      int p = bc->recvproc[i][d];
      if (p >= 0) {	
	MPI_Irecv(bc_rbuff[i][d], bc->nrecv[i][d]*nin,  MPI_DOUBLE, p,
		  d + 1000 * i,  bc->comm, &recvreq[i][d]);
	bc_recv_done[i][d]=0; 
      } else {	
	bc_recv_done[i][d]=1; 
      }
    }
  }
  for (int i = 0; i < 3; i++) {
    int maxsendrecv=MAX(MAX(bc->nsend[i][0],bc->nsend[i][1]),
			MAX(bc->nrecv[i][0],bc->nrecv[i][1]))*nin*sizeof(double);
    if (maxsendrecv<GPAW_CUDA_JOIN_SIZE &&
	bc->recvproc[i][0]>=0 && bc->recvproc[i][1]>=0) {
      bc_join[i]=1;
    } else {
      bc_join[i]=0;
    }
    if (maxsendrecv<GPAW_CUDA_ASYNC_SIZE) {
      bc_async[i]=0;      
    } else {
      bc_async[i]=1;      
    }
  }
}

void bc_unpack_cuda_gpu_async(const boundary_conditions* bc,
				  const double* aa1, double* aa2, int i,
				  MPI_Request recvreq[3][2],
				  MPI_Request sendreq[2],
				  const double_complex phases[2], 
				  cudaStream_t kernel_stream,
				  int nin)

{
  
  bool real = (bc->ndouble == 1);

  bc_alloc_buffers(bc,nin);
  int rank;
  
#ifdef PARALLEL
  
  // Prepare send-buffers
  int send_done=0;
  
  if (bc->sendproc[i][0]>=0 || bc->sendproc[i][1]>=0) {
    for (int d = 0; d < 2; d++) {
      sendreq[d] = 0;
      if (bc->sendproc[i][d] >= 0) {
	const int* start = bc->sendstart[i][d];
	const int* size = bc->sendsize[i][d];
	if (real)
	  bmgs_cut_cuda_gpu(aa2, bc->size2, start,
			    bc_sbuff_gpu[i][d],
			    size,nin,kernel_stream);
	else {
	  cuDoubleComplex phase={creal(phases[d]),cimag(phases[d])};
	  bmgs_cut_cuda_gpuz((cuDoubleComplex*)(aa2), bc->size2, 
			     start,
			     (cuDoubleComplex*)(bc_sbuff_gpu[i][d]),
			     size, phase, nin, kernel_stream);
	}
      }
    }
    GPAW_CUDAMEMCPY_A(bc_sbuff[i][0], bc_sbuff_gpu[i][0],double,
		      (bc->nsend[i][0]+bc->nsend[i][1])*nin,
		      cudaMemcpyDeviceToHost,kernel_stream);    
    cudaEventRecord(bc_sendcpy_event[i],kernel_stream);      
  }


  if (!(bc->sendproc[i][0]>=0) && !(bc->sendproc[i][1]>=0))
    send_done=1;
  
  int ddd[3]={1,1,1};
  for (int ii=i;ii<3;ii++) 
    if (bc_recv_done[ii][ddd[ii]])
      ddd[ii]=1-ddd[ii];
  
  do {
    if (!send_done && cudaEventQuery(bc_sendcpy_event[i])==cudaSuccess) {
      for (int d=0;d<2;d++){
	if (bc->sendproc[i][d]>=0) {
	  MPI_Isend(bc_sbuff[i][d], 
		    bc->nsend[i][d]*nin , MPI_DOUBLE, 
		    bc->sendproc[i][d],
		    1 - d + 1000 * i, bc->comm, 
		    &sendreq[d]);	  
	}	
      }
      send_done=1;
    }
    for (int ii=i;ii<MIN(i+2,3);ii++) {
      int iii=ii;
      if (ii>i && ii==1 && bc_recv_done[ii][0] && bc_recv_done[ii][1])  
	iii=2;
      if (!bc_recv_done[iii][ddd[iii]]) {
	int status;
	MPI_Test(&recvreq[iii][ddd[iii]],&status, MPI_STATUS_IGNORE);
	if (status){
	  int status2=0;
	  if (!bc_recv_done[iii][1-ddd[iii]]) 
	    MPI_Test(&recvreq[iii][1-ddd[iii]],&status2, MPI_STATUS_IGNORE);
	  if (status2) {	  
	    GPAW_CUDAMEMCPY_A(bc_rbuff_gpu[iii][0], bc_rbuff[iii][0],double,
			      (bc->nrecv[iii][0]+bc->nrecv[iii][1])*nin,
			      cudaMemcpyHostToDevice,
			      bc_recv_stream);
	    for (int d = 0; d < 2; d++) {
	      if (real)
		bmgs_paste_cuda_gpu(bc_rbuff_gpu[iii][d], bc->recvsize[iii][d],
				    aa2, bc->size2, bc->recvstart[iii][d],nin, 
				    bc_recv_stream);
	      
	      else
		bmgs_paste_cuda_gpuz((const cuDoubleComplex*)(bc_rbuff_gpu[iii][d]),
				     bc->recvsize[iii][d],
				     (cuDoubleComplex*)(aa2),
				     bc->size2, bc->recvstart[iii][d],nin, 
				     bc_recv_stream); 
	      cudaEventRecord(bc_recv_event[iii][d],bc_recv_stream);
	      bc_recv_done[iii][d]=1; 
	      
	    }
	  } else if (!bc_join[iii]) {
	    GPAW_CUDAMEMCPY_A(bc_rbuff_gpu[iii][ddd[iii]], bc_rbuff[iii][ddd[iii]],
			      double,(bc->nrecv[iii][ddd[iii]])*nin,
			      cudaMemcpyHostToDevice,
			      bc_recv_stream);
	  if (real)
	    bmgs_paste_cuda_gpu(bc_rbuff_gpu[iii][ddd[iii]], 
				bc->recvsize[iii][ddd[iii]],
				aa2, bc->size2, bc->recvstart[iii][ddd[iii]],nin, 
				bc_recv_stream);
	  
	  else
	    bmgs_paste_cuda_gpuz((const cuDoubleComplex*)(bc_rbuff_gpu[iii][ddd[iii]]),
				 bc->recvsize[iii][ddd[iii]],
				 (cuDoubleComplex*)(aa2),
				 bc->size2, bc->recvstart[iii][ddd[iii]],nin, 
				 bc_recv_stream); 
	  cudaEventRecord(bc_recv_event[iii][ddd[iii]],bc_recv_stream);
	  bc_recv_done[iii][ddd[iii]]=1; 
	  }
	}
      }
      if (bc_recv_done[i][0] && bc_recv_done[i][1] && send_done) break;
      ddd[iii]=1-ddd[iii];
      if (bc_recv_done[iii][ddd[iii]])
	ddd[iii]=1-ddd[iii];
    }
  } while(!bc_recv_done[i][0] || !bc_recv_done[i][1] || !send_done); 
  
#endif // Parallel
  // Copy data for periodic boundary conditions:
  for (int d = 0; d < 2; d++) {
    if (bc->sendproc[i][d] == COPY_DATA)  {      
      if (real) {
	bmgs_translate_cuda_gpu(aa2, bc->size2, bc->sendsize[i][d],
				bc->sendstart[i][d], bc->recvstart[i][1 - d],
				nin,kernel_stream);
      } else {
	cuDoubleComplex phase={creal(phases[d]),cimag(phases[d])};
	bmgs_translate_cuda_gpuz((cuDoubleComplex*)(aa2), 
				 bc->size2,bc->sendsize[i][d],
				 bc->sendstart[i][d],bc->recvstart[i][1 - d],
				 phase,nin,kernel_stream);
      }
    }
  }
#ifdef PARALLEL
  // This does not work on the ibm with gcc!  We do a blocking send instead.
  
  for (int d = 0; d < 2; d++){
    if (bc->sendproc[i][d] >= 0)
      assert(MPI_Wait(&sendreq[d],  MPI_STATUS_IGNORE)==MPI_SUCCESS);
  }
  for (int d = 0; d < 2; d++) {
    if (bc->recvproc[i][d]>=0)     
      cudaStreamWaitEvent(kernel_stream,bc_recv_event[i][d],0);
  }

#endif // Parallel  
}

void bc_unpack_cuda_gpu(const boundary_conditions* bc,
			const double* aa1, double* aa2, int i,
			MPI_Request recvreq[3][2],
			MPI_Request sendreq[2],
			const double_complex phases[2], 
			cudaStream_t kernel_stream,
			int nin)
{
    if (!bc_async[i]) {
      bc_unpack_cuda_gpu_sync(bc, aa1, aa2, i, recvreq, sendreq,
			    phases,kernel_stream,nin);
    }  else {
      bc_unpack_cuda_gpu_async(bc, aa1, aa2, i, recvreq, sendreq,
			       phases,kernel_stream,nin);
    }
}


