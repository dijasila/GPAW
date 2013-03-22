
__device__ unsigned int INNAME(lfc_retirementCount) = {0};

#define LFC_REDUCE_MAX_NVEC 16

__global__ void INNAME(integrate_mul_kernel)(const Tcuda *a_G,int nG,
					     const LFVolume_gpu *volume_W, 
					     const int *volume_WMi_gpu,
					     const int *WMi_gpu, 
					     int WMimax,
					     int q, 
					     Tcuda *out,int block_out,
					     Tcuda *results,int Mcount,int nM, int nvec)
  
{

  int yy=gridDim.y/Mcount;

  int bloy=blockIdx.y/yy;
  int block=blockIdx.y-bloy*yy;
  
  unsigned int tid = threadIdx.x;
  unsigned int gridSize = REDUCE_LFC_THREADS*gridDim.x;

  extern __shared__ Tcuda Zcuda(sdata)[];

  // __shared__ Tcuda a_Gvsbuf[REDUCE_LFC_THREADS][LFC_REDUCE_MAX_NVEC];

  Tcuda a_Gvs[LFC_REDUCE_MAX_NVEC];
  //Tcuda a_Gvs[nvec];

  //Tcuda *a_Gvs=a_Gvsbuf[tid];
      
  // perform first level of reduction,
  // reading from global memory, writing to shared memory

  a_G+=nG*block;
  
  for (int vv=0;vv<WMi_gpu[bloy];vv++) {
    const LFVolume_gpu *v=&volume_W[volume_WMi_gpu[bloy*WMimax+vv]];

    //int nB=v->nB;
    
    //int *GB1=v->GB1;
    int *nGBcum=v->nGBcum;
    
    int len_A_gm=v->len_A_gm;
    /*#ifdef CUGPAWCOMPLEX
    cuDoubleComplex phase=v->phase_k[q]; 
    #endif*/
    //double *A_gm=v->A_gm;
5~    //   int nm=v->nm;
    Tcuda *out_t=out+v->M*block_out+block*nM*block_out;
    //Tcuda a_Gv = MAKED(0);

    unsigned int i_b = blockIdx.x*(REDUCE_LFC_THREADS) + tid;
    if (len_A_gm <= gridSize){
      if (i_b < len_A_gm)    {	  
	int c=v->nB*i_b/len_A_gm+1;
	if (nGBcum[c]<=i_b) {
	  while(nGBcum[++c]<=i_b);
	  c--;
	} else {
	  while(nGBcum[--c]>i_b);
	}
	c=v->GB1[c]+i_b-nGBcum[c];

#ifdef CUGPAWCOMPLEX
	/*
	switch (nvec) {
	case 16:
	  a_Gvs[15]=MULTT(a_G[c+15*nG], phase);
	case 15:
	  a_Gvs[14]=MULTT(a_G[c+14*nG], phase);
	case 14:
	  a_Gvs[13]=MULTT(a_G[c+13*nG], phase);
	case 13:
	  a_Gvs[12]=MULTT(a_G[c+12*nG], phase);
	case 12:
	  a_Gvs[11]=MULTT(a_G[c+11*nG], phase);
	case 11:
	  a_Gvs[10]=MULTT(a_G[c+10*nG], phase);
	case 10:
	  a_Gvs[9]=MULTT(a_G[c+9*nG], phase);
	case 9:
	  a_Gvs[8]=MULTT(a_G[c+8*nG], phase);
	case 8:
	  a_Gvs[7]=MULTT(a_G[c+7*nG], phase);
	case 7:
	  a_Gvs[6]=MULTT(a_G[c+6*nG], phase);
	case 6:
	  a_Gvs[5]=MULTT(a_G[c+5*nG], phase);
	case 5:
	  a_Gvs[4]=MULTT(a_G[c+4*nG], phase);
	case 4:
	  a_Gvs[3]=MULTT(a_G[c+3*nG], phase);
	case 3:
	  a_Gvs[2]=MULTT(a_G[c+2*nG], phase);
	case 2:
	  a_Gvs[1]=MULTT(a_G[c+1*nG], phase);
	case 1:
	  a_Gvs[0]=MULTT(a_G[c+0*nG], phase);
	  }*/
	for (int i=0;i<nvec;i++) {
	  a_Gvs[i]=MULTT(a_G[c+i*nG], v->phase_k[q]);
	  //a_Gv=MULTT(a_G[c+i*nG], v->phase_k[q]);
	}
#else
/*
	switch (nvec) {
	case 16:
	  a_Gvs[15]=a_G[c+15*nG];	  
	case 15:
	  a_Gvs[14]=a_G[c+14*nG];	  
	case 14:
	  a_Gvs[13]=a_G[c+13*nG];	  
	case 13:
	  a_Gvs[12]=a_G[c+12*nG];	  
	case 12:
	  a_Gvs[11]=a_G[c+11*nG];	  
	case 11:
	  a_Gvs[10]=a_G[c+10*nG];	  
	case 10:
	  a_Gvs[9]=a_G[c+9*nG];	  
	case 9:
	  a_Gvs[8]=a_G[c+8*nG];	  
	case 8:
	  a_Gvs[7]=a_G[c+7*nG];	  
	case 7:
	  a_Gvs[6]=a_G[c+6*nG];	  
	case 6:
	  a_Gvs[5]=a_G[c+5*nG];	  
	case 5:
	  a_Gvs[4]=a_G[c+4*nG];	  
	case 4:
	  a_Gvs[3]=a_G[c+3*nG];	  
	case 3:
	  a_Gvs[2]=a_G[c+2*nG];	  
	case 2:
	  a_Gvs[1]=a_G[c+1*nG];	  
	case 1:
	  a_Gvs[0]=a_G[c+0*nG];	  
	  }*/
	for (int i=0;i<nvec;i++) {
	  a_Gvs[i]=a_G[c+i*nG];
	  //a_Gv=a_G[c+i*nG];
	}
#endif
	
      }
    }
    for (int m = 0; m < v->nm; m++){  
      double A_gmv;
      if (i_b<len_A_gm)
	A_gmv=v->A_gm[i_b+m*len_A_gm];
      for (int i=0;i<nvec;i++) {
	Tcuda mySum = MAKED(0);
	
	if (len_A_gm <= gridSize){	
	  if (i_b<len_A_gm){
	    /*
	    switch (i) {
	    case 15:
	      mySum=MULTD(a_Gvs[15],A_gmv); break;
	    case 14:
	      mySum=MULTD(a_Gvs[14],A_gmv); break;
	    case 13:
	      mySum=MULTD(a_Gvs[13],A_gmv); break;
	    case 12:
	      mySum=MULTD(a_Gvs[12],A_gmv); break;
	    case 11:
	      mySum=MULTD(a_Gvs[11],A_gmv); break;
	    case 10:
	      mySum=MULTD(a_Gvs[10],A_gmv); break;
	    case 9:
	      mySum=MULTD(a_Gvs[9],A_gmv); break;
	    case 8:
	      mySum=MULTD(a_Gvs[8],A_gmv); break;
	    case 7:
	      mySum=MULTD(a_Gvs[7],A_gmv); break;
	    case 6:
	      mySum=MULTD(a_Gvs[6],A_gmv); break;
	    case 5:
	      mySum=MULTD(a_Gvs[5],A_gmv); break;
	    case 4:
	      mySum=MULTD(a_Gvs[4],A_gmv); break;
	    case 3:
	      mySum=MULTD(a_Gvs[3],A_gmv); break;
	    case 2:
	      mySum=MULTD(a_Gvs[2],A_gmv); break;
	    case 1:
	      mySum=MULTD(a_Gvs[1],A_gmv); break;
	    case 0:
	      mySum=MULTD(a_Gvs[0],A_gmv); break;
	    }*/
	    mySum=MULTD(a_Gvs[i],A_gmv);
	    //mySum=MULTD(a_Gv,A_gmv);
	  }
	  //if (i_b<len_A_gm) mySum=MULTD(a_Gv,A_gm[i_b+m*len_A_gm]);
	} else {
	  unsigned int i_bb=i_b;
	  while (i_bb < len_A_gm)    {	  	 
	    int c=v->nB*i_bb/len_A_gm+1;
	    if (nGBcum[c]<=i_bb) {
	      while(nGBcum[++c]<=i_bb);
	      c--;
	    } else {
	      while(nGBcum[--c]>i_bb);
	    }
#ifdef CUGPAWCOMPLEX	    
	    IADD(mySum,MULTD(MULTT(a_G[v->GB1[c]+i_bb-nGBcum[c]+i*nG],v->phase_k[q]),v->A_gm[i_bb+m*len_A_gm]));
#else
	    IADD(mySum,MULTD(a_G[v->GB1[c]+i_bb-nGBcum[c]+i*nG],v->A_gm[i_bb+m*len_A_gm]));
#endif	    
	    i_bb += gridDim.x*REDUCE_LFC_THREADS;
	  }
	}
	Zcuda(sdata)[tid] = mySum;
	__syncthreads();
	
	if (REDUCE_LFC_THREADS >= 512) { 
	  if (tid < 256) { 
	    Zcuda(sdata)[tid] = mySum = ADD(mySum,Zcuda(sdata)[tid + 256]);
	  }
	  __syncthreads(); 
	}
	if (REDUCE_LFC_THREADS >= 256) { 
	  if (tid < 128) { 
	    Zcuda(sdata)[tid] = mySum = ADD(mySum,Zcuda(sdata)[tid + 128]);
	  } 
	  __syncthreads(); 
	}
	if (REDUCE_LFC_THREADS >= 128) { 
	  if (tid <  64) { 
	    Zcuda(sdata)[tid] = mySum = ADD(mySum,Zcuda(sdata)[tid +  64]);
	  }
	  __syncthreads(); 
	}
	
	if (tid < 32)
	  {
	    volatile Tcuda *smem = Zcuda(sdata);
#ifdef CUGPAWCOMPLEX	
	    if (REDUCE_LFC_THREADS >=  64){  
	      smem[tid].x = mySum.x = mySum.x + smem[tid + 32].x;
	      smem[tid].y = mySum.y = mySum.y + smem[tid + 32].y;
	    }
	    if (REDUCE_LFC_THREADS >=  32){  
	      smem[tid].x = mySum.x = mySum.x + smem[tid + 16].x;
	      smem[tid].y = mySum.y = mySum.y + smem[tid + 16].y;
	    }
	    if (REDUCE_LFC_THREADS >=  16){  
	      smem[tid].x = mySum.x = mySum.x + smem[tid + 8].x;
	      smem[tid].y = mySum.y = mySum.y + smem[tid + 8].y;
	    }
	    if (REDUCE_LFC_THREADS >=  8){  
	      smem[tid].x = mySum.x = mySum.x + smem[tid + 4].x;
	      smem[tid].y = mySum.y = mySum.y + smem[tid + 4].y;
	    }
	    if (REDUCE_LFC_THREADS >=  4){  
	      smem[tid].x = mySum.x = mySum.x + smem[tid + 2].x;
	      smem[tid].y = mySum.y = mySum.y + smem[tid + 2].y;
	    }
	    if (REDUCE_LFC_THREADS >=  2){  
	      smem[tid].x = mySum.x = mySum.x + smem[tid + 1].x;
	      smem[tid].y = mySum.y = mySum.y + smem[tid + 1].y;	
	    }
#else
	    if (REDUCE_LFC_THREADS >=  64)  
	      smem[tid] = mySum = ADD(mySum , smem[tid + 32]);
	    if (REDUCE_LFC_THREADS >=  32)  
	      smem[tid] = mySum = ADD(mySum , smem[tid + 16]);
	    if (REDUCE_LFC_THREADS >=  16)  
	      smem[tid] = mySum = ADD(mySum , smem[tid + 8]);
	    if (REDUCE_LFC_THREADS >=  8)  
	      smem[tid] = mySum = ADD(mySum , smem[tid + 4]);
	    if (REDUCE_LFC_THREADS >=  4)  
	      smem[tid] = mySum = ADD(mySum , smem[tid + 2]);
	    if (REDUCE_LFC_THREADS >=  2)  
	      smem[tid] = mySum = ADD(mySum , smem[tid + 1]);	
#endif
	  }  
	
	
	// write result for this block to global mem
	
	if (tid==0) {
	  if (vv==0)
	    out_t[blockIdx.x+m*block_out+i*nM*block_out] = Zcuda(sdata)[0];
	  else
	    IADD(out_t[blockIdx.x+m*block_out+i*nM*block_out], Zcuda(sdata)[0]);
	}        
	__syncthreads();                
      }
    }
  }
  
  if (gridDim.x==1){
    __shared__ bool amLast;
    __threadfence();
    if (tid == 0) {
      unsigned int ticket = atomicInc(&INNAME(lfc_retirementCount), gridDim.y);
      amLast = (ticket == gridDim.y-1);
    }
    __syncthreads();
    if ((amLast) ) {
      for (int i=tid;i<nM*yy*nvec;i+=blockDim.x){
	results[i]=out[i*block_out];
      }  
      INNAME(lfc_retirementCount)=0;

    }
    
  }
  
  
}

