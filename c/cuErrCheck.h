// Define this to turn on error checking
#define CUDA_ERROR_CHECK
 
#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define CublasSafeCall( err ) __cublasSafeCall( err, __FILE__, __LINE__ )
#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )
 
inline void __cudaSafeCall( cudaError_t err, const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        while (1) sleep(10000000);
    }
#endif
 
    return;
}

inline void __cublasSafeCall( cublasStatus_t err, const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    if ( CUBLAS_STATUS_SUCCESS != err )
    {
        fprintf( stderr, "cublasSafeCall() failed at %s:%i : %i\n",
                 file, line, err);
        exit( -1 );
    }
#endif
 
    return;
}

 
inline void __cudaCheckError( const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    cudaError_t err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s (%i)\n",
                 file, line, cudaGetErrorString( err ), err );
        while (1) sleep(10000000);
    }
 
    // More careful checking. However, this will affect performance.
    // Comment away if needed.
    err = cudaDeviceSynchronize();
    if( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s (%i)\n",
                 file, line, cudaGetErrorString( err ), err );
        while (1) sleep(10000000);
    }
#endif
 
    return;
}
