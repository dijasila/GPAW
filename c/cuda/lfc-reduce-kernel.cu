
__device__ unsigned int INNAME(lfc_retirementCount) = {0};

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
  unsigned int i_b = blockIdx.x*(REDUCE_LFC_THREADS) + tid;

  extern __shared__ Tcuda Zcuda(sdata)[];

  // perform first level of reduction,
  // reading from global memory, writing to shared memory

  a_G+=nG*block;

  for (int vv=0;vv<WMi_gpu[bloy];vv++) {
    const LFVolume_gpu *v=&volume_W[volume_WMi_gpu[bloy*WMimax+vv]];
    int *nGBcum=v->nGBcum;
#ifdef CUGPAWCOMPLEX
    Tcuda phase=v->phase_k[q];
#endif
    
    int len_A_gm=v->len_A_gm;
    Tcuda *out_t=out+v->M*block_out+block*nM*block_out;
    int cc;

    if (len_A_gm <= gridSize){
      if (i_b < len_A_gm)    {	  
	cc=v->nB*i_b/len_A_gm+1;
	if (nGBcum[cc]<=i_b) {
	  while(nGBcum[++cc]<=i_b);
	  cc--;
	} else {
	  while(nGBcum[--cc]>i_b);
	}
	cc=v->GB1[cc]+i_b-nGBcum[cc];
      }
    }
    for (int m = 0; m < v->nm; m++){  
      Tcuda A_gmv;
      const Tcuda *a_G2=a_G;
      Tcuda *out_t2=out_t;
      if (i_b<len_A_gm){
#ifdef CUGPAWCOMPLEX
	A_gmv=MULTD(phase,v->A_gm[i_b+m*len_A_gm]);
#else
	A_gmv=v->A_gm[i_b+m*len_A_gm];
#endif	    
      }
      for (int i=0;i<nvec;i++) {
	Tcuda mySum = MAKED(0);	
	if (len_A_gm <= gridSize){	
	  if (i_b<len_A_gm){
	    mySum=MULTT(a_G2[cc],A_gmv);
	  }
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
	    c=v->GB1[c]+i_bb-nGBcum[c];
#ifdef CUGPAWCOMPLEX	    
	    IADD(mySum,MULTD(MULTT(a_G2[c],phase),v->A_gm[i_bb+m*len_A_gm]));
#else
	    IADD(mySum,MULTD(a_G2[c],v->A_gm[i_bb+m*len_A_gm]));
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
	    out_t2[blockIdx.x] = Zcuda(sdata)[0];
	  else
	    IADD(out_t2[blockIdx.x], Zcuda(sdata)[0]);
	}        
	a_G2+=nG;              
	out_t2+=nM*block_out;              
	__syncthreads();  
      }
      out_t+=block_out;
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

