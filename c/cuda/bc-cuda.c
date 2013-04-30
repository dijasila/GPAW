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


#ifndef CUDA_MPI  
static cudaStream_t bc_recv_stream;
static int bc_streams = 0;
static cudaEvent_t bc_sendcpy_event[3][2];
static cudaEvent_t bc_recv_event[3][2];
#endif // !CUDA_MPI
static int bc_recv_done[3][2];

static int bc_init_count = 0;

#ifndef CUDA_MPI  
static double *bc_rbuff[3][2];
static double *bc_sbuff[3][2];
static double *bc_rbuffs=NULL;
static double *bc_sbuffs=NULL;
#endif // !CUDA_MPI
static double *bc_rbuff_gpu[3][2];
static double *bc_sbuff_gpu[3][2];
static double *bc_rbuffs_gpu=NULL; 
static double *bc_sbuffs_gpu=NULL;
static int bc_rbuffs_size=0;
static int bc_sbuffs_size=0;
static int bc_rbuffs_max=0;
static int bc_sbuffs_max=0;

void bc_init_cuda(boundary_conditions* bc)
{
  int nsends=0;
  int nrecvs=0;

  for (int i=0;i<3;i++)
    for (int d=0;d<2;d++) {
      nsends+=NEXTPITCHDIV(bc->nsend[i][d]);
      nrecvs+=NEXTPITCHDIV(bc->nrecv[i][d]);
    }

  bc_sbuffs_max=MAX(nsends, bc_sbuffs_max);
  bc_rbuffs_max=MAX(nrecvs, bc_rbuffs_max);
  
  bc_init_count++;
}

void bc_init_buffers_cuda()
{
#ifndef CUDA_MPI  
    bc_rbuffs=NULL;
    bc_sbuffs=NULL;
    bc_streams=0;
#endif // !CUDA_MPI
    bc_rbuffs_gpu=NULL;
    bc_sbuffs_gpu=NULL;    
    bc_rbuffs_size=0;
    bc_sbuffs_size=0;
    bc_init_count=0;

}

void bc_alloc_buffers(const boundary_conditions* bc,int blocks)
{
  int nsends=0;
  int nrecvs=0;

  for (int i=0;i<3;i++)
    for (int d=0;d<2;d++) {
      nsends+=NEXTPITCHDIV(bc->nsend[i][d]*blocks);
      nrecvs+=NEXTPITCHDIV(bc->nrecv[i][d]*blocks);
    }

  bc_sbuffs_max=MAX(nsends, bc_sbuffs_max);  
  if (bc_sbuffs_max > bc_sbuffs_size) {
#ifndef CUDA_MPI  
    cudaFreeHost(bc_sbuffs);
    cudaGetLastError();
    GPAW_CUDAMALLOC_HOST(&bc_sbuffs,double, bc_sbuffs_max);
#endif // !CUDA_MPI  
    cudaFree(bc_sbuffs_gpu);
    cudaGetLastError();
    GPAW_CUDAMALLOC(&bc_sbuffs_gpu,double, bc_sbuffs_max);
    bc_sbuffs_size=bc_sbuffs_max;
  }

  bc_rbuffs_max=MAX(nrecvs, bc_rbuffs_max);
  if (bc_rbuffs_max > bc_rbuffs_size) {
#ifndef CUDA_MPI  
    cudaFreeHost(bc_rbuffs);
    cudaGetLastError();
    GPAW_CUDAMALLOC_HOST(&bc_rbuffs,double, bc_rbuffs_max);
#endif // !CUDA_MPI
    cudaFree(bc_rbuffs_gpu);
    cudaGetLastError();
    GPAW_CUDAMALLOC(&bc_rbuffs_gpu,double, bc_rbuffs_max);
    bc_rbuffs_size=bc_rbuffs_max;
  }

#ifndef CUDA_MPI  
  if  (!bc_streams){    
    cudaStreamCreate(&bc_recv_stream);
    bc_streams=1;
    for (int d=0;d<3;d++){
      for (int i=0;i<2;i++){
	cudaEventCreateWithFlags(&bc_sendcpy_event[d][i],
				 cudaEventDefault|cudaEventDisableTiming);
	cudaEventCreateWithFlags(&bc_recv_event[d][i],
				 cudaEventDefault|cudaEventDisableTiming);
      }	
    }
  }
#endif // !CUDA_MPI  
}

void bc_dealloc_cuda(int force)
{
  if (force)
    bc_init_count=1;

  if (bc_init_count==1) {
#ifndef CUDA_MPI  
    cudaError_t rval;
    rval=cudaFreeHost(bc_sbuffs);
    if (rval==cudaSuccess) {
      cudaFreeHost(bc_rbuffs);      
      if  (bc_streams){      
	cudaStreamDestroy(bc_recv_stream);
	for (int d=0;d<3;d++){
	  for (int i=0;i<2;i++){
	    cudaEventDestroy(bc_sendcpy_event[d][i]);  
	    cudaEventDestroy(bc_recv_event[d][i]);
	  }
	}
      }
    }  
#endif // !CUDA_MPI
    
    cudaFree(bc_sbuffs_gpu);
    cudaFree(bc_rbuffs_gpu);
    bc_init_buffers_cuda();
    return;
  }
  if (bc_init_count>0) bc_init_count--;
 
}


void bc_unpack_paste_cuda_gpu(boundary_conditions* bc,
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



#ifndef CUDA_MPI  
  for (int i = 0; i < 3; i++) {    
    int maxrecv=MAX(bc->nrecv[i][0],bc->nrecv[i][1])*nin*sizeof(double);
    bc->cuda_rjoin[i]=0;
    if (bc->recvproc[i][0]>=0 && bc->recvproc[i][1]>=0) {
      if (maxrecv<GPAW_CUDA_RJOIN_SIZE)	
	bc->cuda_rjoin[i]=1;
      else if ((maxrecv<GPAW_CUDA_RJOIN_SAME_SIZE) && 
	       (bc->recvproc[i][0] == bc->recvproc[i][1]))	
	bc->cuda_rjoin[i]=1;
    }    
    int maxsend=MAX(bc->nsend[i][0],bc->nsend[i][1])*nin*sizeof(double);
    bc->cuda_sjoin[i]=0;
    if (bc->sendproc[i][0]>=0 && bc->sendproc[i][1]>=0) {
      if (maxsend<GPAW_CUDA_SJOIN_SIZE)	
	bc->cuda_sjoin[i]=1;
      else if ((maxsend<GPAW_CUDA_SJOIN_SAME_SIZE) && 
	       (bc->sendproc[i][0] == bc->sendproc[i][1]))	
	bc->cuda_sjoin[i]=1;
    }
    
    if (MAX(maxsend,maxrecv)<GPAW_CUDA_ASYNC_SIZE) 
      bc->cuda_async[i]=0;      
    else 
      bc->cuda_async[i]=1;      
  }
#endif // !CUDA_MPI  

  int recvp=0,sendp=0;  
#ifndef CUDA_MPI  
  for (int i=0;i<3;i++){
    bc_sbuff[i][0]=bc_sbuffs+sendp;
    bc_sbuff_gpu[i][0]=bc_sbuffs_gpu+sendp;
    if (!bc->cuda_async[i] || bc->cuda_sjoin[i]){
      bc_sbuff[i][1]=bc_sbuffs+sendp+bc->nsend[i][0]*nin;
      bc_sbuff_gpu[i][1]=bc_sbuffs_gpu+sendp+bc->nsend[i][0]*nin;
      sendp+=NEXTPITCHDIV((bc->nsend[i][0]+bc->nsend[i][1])*nin);
    } else {
      sendp+=NEXTPITCHDIV((bc->nsend[i][0])*nin);
      bc_sbuff[i][1]=bc_sbuffs+sendp;
      bc_sbuff_gpu[i][1]=bc_sbuffs_gpu+sendp;
      sendp+=NEXTPITCHDIV((bc->nsend[i][1])*nin);
    }
    bc_rbuff[i][0]=bc_rbuffs+recvp;
    bc_rbuff_gpu[i][0]=bc_rbuffs_gpu+recvp;
    if (!bc->cuda_async[i] || bc->cuda_rjoin[i]){
      bc_rbuff[i][1]=bc_rbuffs+recvp+bc->nrecv[i][0]*nin;
      bc_rbuff_gpu[i][1]=bc_rbuffs_gpu+recvp+bc->nrecv[i][0]*nin;
      recvp+=NEXTPITCHDIV((bc->nrecv[i][0]+bc->nrecv[i][1])*nin);
    } else {
      recvp+=NEXTPITCHDIV((bc->nrecv[i][0])*nin);
      bc_rbuff[i][1]=bc_rbuffs+recvp;
      bc_rbuff_gpu[i][1]=bc_rbuffs_gpu+recvp;
      recvp+=NEXTPITCHDIV((bc->nrecv[i][1])*nin);
    }
  }
#else // !CUDA_MPI  
  for (int i=0;i<3;i++){
    bc_sbuff_gpu[i][0]=bc_sbuffs_gpu+sendp;
    if (!bc->cuda_async[i] || bc->cuda_sjoin[i]){
      bc_sbuff_gpu[i][1]=bc_sbuffs_gpu+sendp+bc->nsend[i][0]*nin;
      sendp+=NEXTPITCHDIV((bc->nsend[i][0]+bc->nsend[i][1])*nin);
    } else {
      sendp+=NEXTPITCHDIV((bc->nsend[i][0])*nin);
      bc_sbuff_gpu[i][1]=bc_sbuffs_gpu+sendp;
      sendp+=NEXTPITCHDIV((bc->nsend[i][1])*nin);
    }
    bc_rbuff_gpu[i][0]=bc_rbuffs_gpu+recvp;
    if (!bc->cuda_async[i] || bc->cuda_rjoin[i]){
      bc_rbuff_gpu[i][1]=bc_rbuffs_gpu+recvp+bc->nrecv[i][0]*nin;
      recvp+=NEXTPITCHDIV((bc->nrecv[i][0]+bc->nrecv[i][1])*nin);
    } else {
      recvp+=NEXTPITCHDIV((bc->nrecv[i][0])*nin);
      bc_rbuff_gpu[i][1]=bc_rbuffs_gpu+recvp;
      recvp+=NEXTPITCHDIV((bc->nrecv[i][1])*nin);
    }
  }
#endif // !CUDA_MPI  

  for (int i = 0; i < 3; i++) {
    for (int d = 0; d < 2; d++) {
      int p = bc->recvproc[i][d];
      if (p >= 0) {	
#ifndef CUDA_MPI  
	MPI_Irecv(bc_rbuff[i][d], bc->nrecv[i][d]*nin,  MPI_DOUBLE, p,
		  d + 1000 * i,  bc->comm, &recvreq[i][d]);
#else // !CUDA_MPI  
	MPI_Irecv(bc_rbuff_gpu[i][d], bc->nrecv[i][d]*nin,  MPI_DOUBLE, p,
		  d + 1000 * i,  bc->comm, &recvreq[i][d]);
#endif // !CUDA_MPI  
	bc_recv_done[i][d]=0; 
      } else {	
	bc_recv_done[i][d]=1; 
      }
    }
  }
  
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

#ifndef CUDA_MPI  
  if (bc->sendproc[i][0]>=0 || bc->sendproc[i][1]>=0)      
    /*    GPAW_CUDAMEMCPY_A(bc_sbuff[i][0],bc_sbuff_gpu[i][0],double,
		      (bc->nsend[i][0]+bc->nsend[i][1]) *  nin, 
		      cudaMemcpyDeviceToHost,kernel_stream);*/
    GPAW_CUDAMEMCPY(bc_sbuff[i][0],bc_sbuff_gpu[i][0],double,
		    (bc->nsend[i][0]+bc->nsend[i][1])*nin, 
		    cudaMemcpyDeviceToHost);
#endif // !CUDA_MPI  

  //Start sending:
  for (int d = 0; d < 2; d++) {
    sendreq[d] = 0;
    int p = bc->sendproc[i][d];
    if (p >= 0) {
#ifndef CUDA_MPI  
      assert(MPI_Isend(bc_sbuff[i][d], bc->nsend[i][d] * nin, MPI_DOUBLE, p,
		       1 - d + 1000 * i, bc->comm, 
		       &sendreq[d])==MPI_SUCCESS);
#else // !CUDA_MPI  
      cudaStreamSynchronize(kernel_stream);  	
      assert(MPI_Isend(bc_sbuff_gpu[i][d], bc->nsend[i][d] * nin, MPI_DOUBLE, p,
		       1 - d + 1000 * i, bc->comm, 
		       &sendreq[d])==MPI_SUCCESS);
#endif // !CUDA_MPI  
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
#ifndef CUDA_MPI  
    GPAW_CUDAMEMCPY(bc_rbuff_gpu[i][0],bc_rbuff[i][0],double,
		    (bc->nrecv[i][0]+bc->nrecv[i][1])*nin, 
		    cudaMemcpyHostToDevice);
#endif
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




#ifndef CUDA_MPI  

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
  int send_done[2]={0,0};
  
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
	if (!bc->cuda_sjoin[i]) {
	  GPAW_CUDAMEMCPY_A(bc_sbuff[i][d], bc_sbuff_gpu[i][d],double,
			    bc->nsend[i][d]*nin,
			    cudaMemcpyDeviceToHost,kernel_stream);    
	  cudaEventRecord(bc_sendcpy_event[i][d],kernel_stream);      
	}
      }
    }
    if (bc->cuda_sjoin[i]) {
      GPAW_CUDAMEMCPY_A(bc_sbuff[i][0], bc_sbuff_gpu[i][0],double,
			(bc->nsend[i][0]+bc->nsend[i][1])*nin,
			cudaMemcpyDeviceToHost,kernel_stream);    
      cudaEventRecord(bc_sendcpy_event[i][0],kernel_stream);      
    }
  }

  for (int d = 0; d < 2; d++) 
    if (!(bc->sendproc[i][d]>=0))
      send_done[d]=1;
  
  int dd=0;
  if (send_done[dd])
    dd=1;

  //int loopc=MIN(1,3-i);
  int loopc=MIN(2,3-i);
  
  int ddd[loopc];
  for (int ii=0;ii<loopc;ii++) {
    ddd[ii]=1;
    if (bc_recv_done[ii+i][ddd[ii]])
      ddd[ii]=1-ddd[ii];
  }

  do {
    if (!send_done[dd] && cudaEventQuery(bc_sendcpy_event[i][dd])==cudaSuccess) {
      MPI_Isend(bc_sbuff[i][dd], 
		bc->nsend[i][dd]*nin , MPI_DOUBLE, 
		bc->sendproc[i][dd],
		1 - dd + 1000 * i, bc->comm, 
		&sendreq[dd]);	  
      send_done[dd]=1;
      dd=1;
      if (bc->cuda_sjoin[i]) {
	MPI_Isend(bc_sbuff[i][dd], 
		  bc->nsend[i][dd]*nin , MPI_DOUBLE, 
		  bc->sendproc[i][dd],
		  1 - dd + 1000 * i, bc->comm, 
		  &sendreq[dd]);	  
	send_done[dd]=1;
      }
      loopc=1;
    }
    for (int i2=0;i2<loopc;i2++) {
      int i3=i2+i;
      if (i==0 && i2==1 && 
	  bc_recv_done[i3][0] && bc_recv_done[i3][1]) {
	i3=2;
      }
      if (!bc->cuda_async[i3])
	continue;

      if (i2==0 && bc->cuda_rjoin[i3] && 
	  !bc_recv_done[i3][0] && !bc_recv_done[i3][1]) {
	int status;
	MPI_Testall(2,recvreq[i3],&status, MPI_STATUSES_IGNORE);
	if (status) {	  
	  GPAW_CUDAMEMCPY_A(bc_rbuff_gpu[i3][0], bc_rbuff[i3][0],double,
			    (bc->nrecv[i3][0]+bc->nrecv[i3][1])*nin,
			    cudaMemcpyHostToDevice,
			    bc_recv_stream);
	  for (int d = 0; d < 2; d++) {
	    if (!bc_recv_done[i3][d]) {
	      if (real)
		bmgs_paste_cuda_gpu(bc_rbuff_gpu[i3][d], bc->recvsize[i3][d],
				    aa2, bc->size2, bc->recvstart[i3][d],nin, 
				    bc_recv_stream);
	      
	      else
		bmgs_paste_cuda_gpuz((const cuDoubleComplex*)(bc_rbuff_gpu[i3][d]),
				     bc->recvsize[i3][d],
				     (cuDoubleComplex*)(aa2),
				     bc->size2, bc->recvstart[i3][d],nin, 
				     bc_recv_stream); 
	      cudaEventRecord(bc_recv_event[i3][d],bc_recv_stream);
	      bc_recv_done[i3][d]=1; 
	    }	      
	  }
	}
      } else if (!bc_recv_done[i3][ddd[i2]]) {
	int status;
	MPI_Test(&recvreq[i3][ddd[i2]],&status, MPI_STATUS_IGNORE);
	if (status){
	  GPAW_CUDAMEMCPY_A(bc_rbuff_gpu[i3][ddd[i2]], bc_rbuff[i3][ddd[i2]],
			    double,(bc->nrecv[i3][ddd[i2]])*nin,
			    cudaMemcpyHostToDevice,
			    bc_recv_stream);
	  if (real)
	    bmgs_paste_cuda_gpu(bc_rbuff_gpu[i3][ddd[i2]], 
				bc->recvsize[i3][ddd[i2]],
				aa2, bc->size2, bc->recvstart[i3][ddd[i2]],nin, 
				bc_recv_stream);
	  
	  else
	    bmgs_paste_cuda_gpuz((const cuDoubleComplex*)(bc_rbuff_gpu[i3][ddd[i2]]),
				 bc->recvsize[i3][ddd[i2]],
				 (cuDoubleComplex*)(aa2),
				 bc->size2, bc->recvstart[i3][ddd[i2]],nin, 
				 bc_recv_stream); 
	  cudaEventRecord(bc_recv_event[i3][ddd[i2]],bc_recv_stream);
	  bc_recv_done[i3][ddd[i2]]=1; 
	}
      }
      if (!bc_recv_done[i3][1-ddd[i2]])
	ddd[i2]=1-ddd[i2];
    }
  } while(!bc_recv_done[i][0] || !bc_recv_done[i][1] || !send_done[0] || !send_done[1]); 
  
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

#else  // !CUDA_MPI  
void bc_unpack_cuda_gpu_async(const boundary_conditions* bc,
				  const double* aa1, double* aa2, int i,
				  MPI_Request recvreq[3][2],
				  MPI_Request sendreq[2],
				  const double_complex phases[2], 
				  cudaStream_t kernel_stream,
				  int nin)

{
  bc_unpack_cuda_gpu_sync(bc, aa1, aa2, i, recvreq, sendreq,
			  phases,kernel_stream,nin);
}
#endif // !CUDA_MPI  

void bc_unpack_cuda_gpu(const boundary_conditions* bc,
			const double* aa1, double* aa2, int i,
			MPI_Request recvreq[3][2],
			MPI_Request sendreq[2],
			const double_complex phases[2], 
			cudaStream_t kernel_stream,
			int nin)
{
#ifndef CUDA_MPI  
    if (!bc->cuda_async[i]) {
      bc_unpack_cuda_gpu_sync(bc, aa1, aa2, i, recvreq, sendreq,
			    phases,kernel_stream,nin);
    }  else {
      bc_unpack_cuda_gpu_async(bc, aa1, aa2, i, recvreq, sendreq,
			       phases,kernel_stream,nin);
    }
#else // !CUDA_MPI  
    bc_unpack_cuda_gpu_sync(bc, aa1, aa2, i, recvreq, sendreq,
			    phases,kernel_stream,nin);
#endif // !CUDA_MPI  
}


