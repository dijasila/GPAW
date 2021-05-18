#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL GPAW_ARRAY_API
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>

#include <float.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <pthread.h>

#include "gpaw-cuda-int.h"

#define HYBRID_GEMM_SIZE_N_GPU  (512/2)
#define HYBRID_GEMM_SIZE_K_GPU  (512/2)
#define HYBRID_GEMM_SIZE_M_GPU  (512*4)

#define HYBRID_GEMM_SIZE_N_CPU  (HYBRID_GEMM_SIZE_N_GPU/2)
#define HYBRID_GEMM_SIZE_K_CPU  (HYBRID_GEMM_SIZE_K_GPU/2)
#define HYBRID_GEMM_SIZE_M_CPU  (HYBRID_GEMM_SIZE_M_GPU/2)
#define HYBRID_GEMM_GPU_MDIV    (64)
#define HYBRID_GEMM_GPU_NDIV    (16)

#define HYBRID_SYRK_SIZE_N_GPU  (512/2)
#define HYBRID_SYRK_SIZE_K_GPU  (512*4)

#define HYBRID_SYRK_SIZE_N_CPU  (HYBRID_SYRK_SIZE_N_GPU/2)
#define HYBRID_SYRK_SIZE_K_CPU  (HYBRID_SYRK_SIZE_K_GPU/2)
#define HYBRID_SYRK_GPU_KDIV    (16)

#define HYBRID_SYR2K_SIZE_N_GPU  (512/2)
#define HYBRID_SYR2K_SIZE_K_GPU  (512*4)

#define HYBRID_SYR2K_SIZE_N_CPU  (HYBRID_SYR2K_SIZE_N_GPU/2)
#define HYBRID_SYR2K_SIZE_K_CPU  (HYBRID_SYR2K_SIZE_K_GPU/2)
#define HYBRID_SYR2K_GPU_KDIV    (16)

#define HYBRID_MAX_PACE  (16)

#define HYBRID_FUNC_MAX_TIMES      (6)


cublasHandle_t _gpaw_cublas_handle;

extern "C" {

typedef struct _hybrid_pace_t {
    unsigned int times;
    unsigned int id;
    double gpu, cpu, dtoh, htod;
} hybrid_pace_t;

typedef struct _hybrid_params_t {
    int init;
    double *a, *b, *c;
    double *c_gpu;
    int size_a, size_b, size_c, size_c_gpu;
    cudaStream_t stream[3];
} hybrid_params_t;

typedef struct _hybrid_func_params_t {
    int init;
    hybrid_pace_t bench;
    int hybrid;
    int k, n, m, n1, n2, m1, m2, k1, k2;
    int beta_null;
    int ndouble;
    cudaEvent_t event_gpu[4];   /* on GPU device */
    cudaEvent_t event_dtoh[2];  /* device -> host */
    cudaEvent_t event_htod[2];  /* host -> device */
    hybrid_pace_t pace[HYBRID_MAX_PACE];
} hybrid_func_params_t;


static hybrid_params_t hybrid_params = {
    .init = 0,
    .a = NULL, .b = NULL, .c = NULL, .c_gpu = NULL,
    .size_a = 0, .size_b = 0, .size_c = 0, .size_c_gpu = 0,
};

static hybrid_func_params_t hybrid_gemm_params = {
    .init = 0,
    .bench = {0,0,0,0,0,0},
    .hybrid = 0,
    .k = 0, .n = 0, .m = 0,
    .n1 = 0, .n2 = 0,
    .m1 = 0, .m2 = 0,
    .k1 = 0, .k2 = 0,
};

static hybrid_func_params_t hybrid_syrk_params = {
    .init = 0,
    .bench = {0,0,0,0,0,0},
    .hybrid = 0,
    .k = 0, .n = 0, .m = 0,
    .n1 = 0, .n2 = 0,
    .m1 = 0, .m2 = 0,
    .k1 = 0, .k2 = 0,
};

static hybrid_func_params_t hybrid_syr2k_params = {
    .init = 0,
    .bench = {0,0,0,0,0,0},
    .hybrid = 0,
    .k = 0, .n = 0, .m = 0,
    .n1 = 0, .n2 = 0,
    .m1 = 0, .m2 = 0,
    .k1 = 0, .k2 = 0,
};


#ifdef GPAW_NO_UNDERSCORE_BLAS
#  define dsyrk_  dsyrk
#  define zherk_  zherk
#  define dsyr2k_ dsyr2k
#  define zher2k_ zher2k
#  define dgemm_  dgemm
#  define zgemm_  zgemm
#endif

void dgemm_(const char *transa, const char *transb, int *m, int *n, int *k,
            double *alpha, double *a, int *lda, double *b, int *ldb,
            double *beta, double *c, int *ldc);

void zgemm_(const char *transa, const char *transb, int *m, int *n, int *k,
            void *alpha, void *a, int *lda, void *b, int *ldb,
            void *beta, void *c, int *ldc);

void dsyrk_(const char *uplo, const char *trans, int *n, int *k,
            double *alpha, double *a, int *lda,
            double *beta, double *c, int *ldc);

void zherk_(const char *uplo, const char *trans, int *n, int *k,
            double *alpha, void *a, int *lda,
            double *beta, void *c, int *ldc);

void dsyr2k_(const char *uplo, const char *trans, int *n, int *k,
             double *alpha, double *a, int *lda, double *b, int *ldb,
             double *beta, double *c, int *ldc);

void zher2k_(const char *uplo, const char *trans, int *n, int *k,
             void *alpha, void *a, int *lda, void *b, int *ldb,
             double *beta, void *c, int *ldc);
}


extern "C"
void blas_init_cuda()
{
    gpaw_cubSCall(cublasCreate(&_gpaw_cublas_handle));
}


inline unsigned int mylog2(unsigned int v)
{
    int r = 0;
    while (v >>= 1)
        r++;
    return r;
}


inline hybrid_pace_t *hybrid_pace_get(hybrid_pace_t *paces, int count,
                                      unsigned int m, unsigned int k,
                                      unsigned int n, unsigned int p)
{
    unsigned int key;
    key = mylog2(m) + (mylog2(k) << 5) + (mylog2(n) << 10)
        + (mylog2(p) << 15);

    for (int i=0; i < count; i++) {
        if (paces[i].id == 0) {
            paces[i].id = key;
            return &paces[i];
        } else if (paces[i].id == key) {
            return &paces[i];
        }
    }
    return &paces[HYBRID_MAX_PACE - 1];
}


extern "C"
void hybrid_pace_init(hybrid_pace_t *paces, int n)
{
    for (int i=0; i < n; i++) {
        paces[i].times = 0;
        paces[i].cpu = 0;
        paces[i].gpu = 0;
        paces[i].dtoh = 0;
        paces[i].htod = 0;
        paces[i].id = 0;
    }
}


extern "C"
void hybrid_pace_update(hybrid_pace_t *pace, double time_gpu,
                        double time_cpu, double time_dtoh,
                        double time_htod)
{
    pace->gpu = (pace->gpu * (pace->times - 1) + time_gpu) / pace->times;
    pace->cpu = (pace->cpu * (pace->times - 1) + time_cpu) / pace->times;
    pace->dtoh = (pace->dtoh * (pace->times - 1) + time_dtoh) / pace->times;
    pace->htod = (pace->htod * (pace->times - 1) + time_htod) / pace->times;
}


extern "C"
void hybrid_func_init(hybrid_func_params_t *pg)
{
    for (int i=0; i < 2; i++) {
        cudaEventCreate(&pg->event_dtoh[i]);
        cudaEventCreate(&pg->event_htod[i]);
    }
    for (int i=0; i < 4; i++) {
        cudaEventCreate(&pg->event_gpu[i]);
    }
    hybrid_pace_init(pg->pace, HYBRID_MAX_PACE);
    pg->init = 1;
}


extern "C"
static void hybrid_param_alloc(hybrid_params_t *ph, int size_a,
                               int size_b, int size_c, int size_c_gpu)
{
    if (!ph->init) {
        for (int i=0; i < 3; i++)
            cudaStreamCreate(&ph->stream[i]);
        if (!ph->a)
            cudaFreeHost(ph->a);
        ph->a = NULL;
        ph->size_a = 0;
        if (!ph->b)
            cudaFreeHost(ph->b);
        ph->b = NULL;
        ph->size_b = 0;
        if (!ph->c)
            cudaFreeHost(ph->c);
        ph->c = NULL;
        ph->size_c = 0;
        if (!ph->a)
            cudaFree(ph->c_gpu);
        ph->c_gpu = NULL;
        ph->size_c_gpu = 0;
        ph->init = 1;
    }
    if (ph->size_a < size_a) {
        if (!ph->a)
            cudaFreeHost(ph->a);
        ph->size_a = size_a;
        GPAW_CUDAMALLOC_HOST(&ph->a, double, ph->size_a);
    }
    if (ph->size_b < size_b) {
        if (!ph->b)
            cudaFreeHost(ph->b);
        ph->size_b = size_b;
        GPAW_CUDAMALLOC_HOST(&ph->b, double, ph->size_b);

    }
    if (ph->size_c < size_c) {
        if (!ph->c)
            cudaFreeHost(ph->c);
        ph->size_c = size_c;
        GPAW_CUDAMALLOC_HOST(&ph->c, double, ph->size_c);
    }
    if (ph->size_c_gpu < size_c_gpu) {
        if (!ph->c_gpu)
            cudaFree(ph->c_gpu);
        ph->size_c_gpu = size_c_gpu;
        GPAW_CUDAMALLOC(&ph->c_gpu, double, ph->size_c_gpu);
    }
}


extern "C"
static void hybrid_gemm_benchmark(hybrid_func_params_t *pg,
                                  hybrid_params_t *ph)
{
    int n1 = HYBRID_GEMM_SIZE_N_GPU;
    int k1 = HYBRID_GEMM_SIZE_K_GPU;
    int m1 = HYBRID_GEMM_SIZE_M_GPU;
    int lda1 = m1;
    int ldb1 = k1;
    int ldc1 = m1;

    int n2 = HYBRID_GEMM_SIZE_N_CPU;
    int k2 = HYBRID_GEMM_SIZE_K_CPU;
    int m2 = HYBRID_GEMM_SIZE_M_CPU;
    int lda2 = m2;
    int ldb2 = k2;
    int ldc2 = m2;

    double *a_gpu, *b_gpu;
    int times = 2;
    float time_gpu, time_cpu, time_dtoh, time_htod;
    double alpha = 1.5;
    double beta = 0.0;

    hybrid_pace_init(&pg->bench, 1);

    GPAW_CUDAMALLOC(&a_gpu, double, k1 * m1);
    GPAW_CUDAMALLOC(&b_gpu, double, k1 * n1);

    hybrid_param_alloc(ph, k2 * m2, k2 * n2, n2 * m2, n1 * m1);

    for (int i=0; i < times+1; i++) {
        gpaw_cubSCall(
                cublasSetStream(_gpaw_cublas_handle, ph->stream[0]));
        /* DGEMM on GPU device */
        cudaEventRecord(pg->event_gpu[0], ph->stream[0]);
        gpaw_cubSCall(
                cublasDgemm(_gpaw_cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                            m1, n1, k1, &alpha, (double*) a_gpu, lda1,
                            (double*) b_gpu, ldb1,
                            &beta, (double*) ph->c_gpu, ldc1));
        cudaEventRecord(pg->event_gpu[1], ph->stream[0]);
        /* data transfer device -> host */
        cudaEventRecord(pg->event_dtoh[0], ph->stream[1]);
        gpaw_cubSCall(
                cublasGetMatrixAsync(m2, k2, sizeof(double),
                                     (void*) a_gpu, m2, (void*) ph->a, m2,
                                     ph->stream[1]));
        cudaEventRecord(pg->event_dtoh[1], ph->stream[1]);

        gpaw_cudaSafeCall(
                cudaStreamSynchronize(ph->stream[1]));
        /* DGEMM on host CPU */
        Py_BEGIN_ALLOW_THREADS;
        dgemm_("n", "n", &m2, &n2, &k2, &alpha, ph->a, &lda2, ph->b, &ldb2,
                &beta, ph->c, &ldc2);
        Py_END_ALLOW_THREADS;
        /* data transfer host -> device */
        cudaEventRecord(pg->event_htod[0], ph->stream[1]);
        gpaw_cubSCall(
                cublasSetMatrixAsync(m2, n2, sizeof(double),
                                     (void*) ph->c, m2, (void*) ph->c_gpu, m2,
                                     ph->stream[1]));
        cudaEventRecord(pg->event_htod[1], ph->stream[1]);

        gpaw_cudaSafeCall(cudaStreamSynchronize(ph->stream[0]));
        gpaw_cudaSafeCall(cudaStreamSynchronize(ph->stream[1]));
        /* elapsed time for each part */
        cudaEventElapsedTime(&time_gpu, pg->event_gpu[0], pg->event_gpu[1]);
        cudaEventElapsedTime(&time_cpu, pg->event_dtoh[1], pg->event_htod[0]);
        cudaEventElapsedTime(&time_dtoh, pg->event_dtoh[0], pg->event_dtoh[1]);
        cudaEventElapsedTime(&time_htod, pg->event_htod[0], pg->event_htod[1]);
        if (i > 0) {
            pg->bench.gpu += time_gpu / ((double) m1 * n1 * k1);
            pg->bench.cpu += time_cpu / ((double) m2 * n2 * k2);
            pg->bench.dtoh += time_dtoh / ((double) (m2 * k2));
            pg->bench.htod += time_htod / ((double) (n2 * m2));
        }
    }
    /* use average over all trials */
    pg->bench.gpu /= times;
    pg->bench.cpu /= times;
    pg->bench.cpu *= 1.5;
    pg->bench.dtoh /= times;
    pg->bench.htod /= times;
    pg->bench.times=1;

    gpaw_cubSCall(cublasSetStream(_gpaw_cublas_handle, 0));
    cudaFree(a_gpu);
    cudaFree(b_gpu);
}


extern "C"
static void hybrid_syrk_benchmark(hybrid_func_params_t *ps,
                                  hybrid_params_t *ph)
{
    int n1 = HYBRID_SYRK_SIZE_N_GPU;
    int k1 = HYBRID_SYRK_SIZE_K_GPU;
    int lda1 = k1;
    int ldc1 = n1;

    int n2 = HYBRID_SYRK_SIZE_N_CPU;
    int k2 = HYBRID_SYRK_SIZE_K_CPU;
    int lda2 = k2;
    int ldc2 = n2;

    double *a_gpu, *b_gpu;
    int times = 2;
    float time_gpu, time_cpu, time_dtoh, time_htod;
    double alpha = 1.5;
    double beta = 0.0;

    hybrid_pace_init(&ps->bench, 1);

    GPAW_CUDAMALLOC(&a_gpu, double, k1 * n1);
    GPAW_CUDAMALLOC(&b_gpu, double, k1 * n1);

    hybrid_param_alloc(ph, k2 * n2, k2 * n2, n2 * n2, n1 * n1);

    for (int i=0; i < times+1; i++) {
        gpaw_cubSCall(
                cublasSetStream(_gpaw_cublas_handle, ph->stream[0]));
        /* DSYRK on GPU device */
        cudaEventRecord(ps->event_gpu[0], ph->stream[0]);
        gpaw_cubSCall(
                cublasDsyrk(_gpaw_cublas_handle, CUBLAS_FILL_MODE_UPPER,
                            CUBLAS_OP_T, n1, k1, &alpha, (double*) a_gpu, lda1,
                            &beta, (double*) ph->c_gpu, ldc1));
        cudaEventRecord(ps->event_gpu[1], ph->stream[0]);
        /* data transfer device -> host */
        cudaEventRecord(ps->event_dtoh[0], ph->stream[1]);
        gpaw_cubSCall(
                cublasGetMatrixAsync(n2, k2, sizeof(double),
                                     (void*) a_gpu, n2, (void*) ph->a, n2,
                                     ph->stream[1]));
        cudaEventRecord(ps->event_dtoh[1], ph->stream[1]);

        gpaw_cudaSafeCall(cudaStreamSynchronize(ph->stream[1]));
        /* DSYRK on host CPU */
        Py_BEGIN_ALLOW_THREADS;
        dsyrk_("u", "t", &n2, &k2, &alpha, ph->a, &lda2, &beta, ph->c, &ldc2);
        Py_END_ALLOW_THREADS;
        /* data transfer host -> device */
        cudaEventRecord(ps->event_htod[0], ph->stream[1]);
        gpaw_cubSCall(
                cublasSetMatrixAsync(n2, n2, sizeof(double),
                                     (void*) ph->c, n2, (void*) ph->c_gpu, n2,
                                     ph->stream[1]));
        cudaEventRecord(ps->event_htod[1], ph->stream[1]);

        gpaw_cudaSafeCall(cudaStreamSynchronize(ph->stream[0]));
        gpaw_cudaSafeCall(cudaStreamSynchronize(ph->stream[1]));
        /* elapsed time for each part */
        cudaEventElapsedTime(&time_gpu, ps->event_gpu[0], ps->event_gpu[1]);
        cudaEventElapsedTime(&time_cpu, ps->event_dtoh[1], ps->event_htod[0]);
        cudaEventElapsedTime(&time_dtoh, ps->event_dtoh[0], ps->event_dtoh[1]);
        cudaEventElapsedTime(&time_htod, ps->event_htod[0], ps->event_htod[1]);
        if (i > 0) {
            ps->bench.gpu += time_gpu / ((double) n1 * n1 * k1);
            ps->bench.cpu += time_cpu / ((double) n2 * n2 * k2);
            ps->bench.dtoh += time_dtoh / ((double) (n2 * k2));
            ps->bench.htod += time_htod / ((double) (n2 * n2));
        }
    }
    /* use average over all trials */
    ps->bench.gpu /= times;
    ps->bench.cpu /= times;
    ps->bench.cpu *= 1.5;
    ps->bench.dtoh /= times;
    ps->bench.htod /= times;
    ps->bench.times = 1;

    gpaw_cubSCall(cublasSetStream(_gpaw_cublas_handle, 0));
    cudaFree(a_gpu);
    cudaFree(b_gpu);
}


extern "C"
static void hybrid_syr2k_benchmark(hybrid_func_params_t *ps,
                                   hybrid_params_t *ph)
{
    int n1 = HYBRID_SYR2K_SIZE_N_GPU;
    int k1 = HYBRID_SYR2K_SIZE_K_GPU;
    int lda1 = k1;
    int ldb1 = k1;
    int ldc1 = n1;

    int n2 = HYBRID_SYR2K_SIZE_N_CPU;
    int k2 = HYBRID_SYR2K_SIZE_K_CPU;
    int lda2 = k2;
    int ldb2 = k2;
    int ldc2 = n2;

    double *a_gpu, *b_gpu;
    int times = 2;
    float time_gpu, time_cpu, time_dtoh, time_htod;
    double alpha = 1.5;
    double beta = 0.0;

    hybrid_pace_init(&ps->bench, 1);

    GPAW_CUDAMALLOC(&a_gpu, double, k1 * n1);
    GPAW_CUDAMALLOC(&b_gpu, double, k1 * n1);

    hybrid_param_alloc(ph, k2 * n2, k2 * n2, n2 * n2, n1 * n1);

    for (int i=0; i < times+1; i++) {
        gpaw_cubSCall(
                cublasSetStream(_gpaw_cublas_handle, ph->stream[0]));
        /* DSYR2K on GPU device */
        cudaEventRecord(ps->event_gpu[0], ph->stream[0]);
        gpaw_cubSCall(
                cublasDsyr2k(_gpaw_cublas_handle, CUBLAS_FILL_MODE_UPPER,
                             CUBLAS_OP_T, n1, k1, &alpha,
                             (double*) a_gpu, lda1, (double*) b_gpu, ldb1,
                             &beta, (double*) ph->c_gpu, ldc1));
        cudaEventRecord(ps->event_gpu[1], ph->stream[0]);
        /* data transfer device -> host */
        cudaEventRecord(ps->event_dtoh[0], ph->stream[1]);
        gpaw_cubSCall(
                cublasGetMatrixAsync(n2, k2, sizeof(double),
                                     (void*) a_gpu, n2, (void*) ph->a, n2,
                                     ph->stream[1]));
        cudaEventRecord(ps->event_dtoh[1], ph->stream[1]);

        gpaw_cudaSafeCall(cudaStreamSynchronize(ph->stream[1]));
        /* DSYR2K on host CPU */
        Py_BEGIN_ALLOW_THREADS;
        dsyr2k_("u", "t", &n2, &k2,
                &alpha, ph->a, &lda2,  ph->b, &ldb2, &beta,
                ph->c, &ldc2);
        Py_END_ALLOW_THREADS;
        /* data transfer host -> device */
        cudaEventRecord(ps->event_htod[0], ph->stream[1]);
        gpaw_cubSCall(
                cublasSetMatrixAsync(n2, n2, sizeof(double),
                                     (void*) ph->c, n2, (void*) ph->c_gpu, n2,
                                     ph->stream[1]));
        cudaEventRecord(ps->event_htod[1], ph->stream[1]);

        gpaw_cudaSafeCall(cudaStreamSynchronize(ph->stream[0]));
        gpaw_cudaSafeCall(cudaStreamSynchronize(ph->stream[1]));
        /* elapsed time for each part */
        cudaEventElapsedTime(&time_gpu, ps->event_gpu[0], ps->event_gpu[1]);
        cudaEventElapsedTime(&time_cpu, ps->event_dtoh[1], ps->event_htod[0]);
        cudaEventElapsedTime(&time_dtoh, ps->event_dtoh[0], ps->event_dtoh[1]);
        cudaEventElapsedTime(&time_htod, ps->event_htod[0], ps->event_htod[1]);
        if (i > 0) {
            ps->bench.gpu += time_gpu / ((double) n1 * n1 * k1);
            ps->bench.cpu += time_cpu / ((double) n2 * n2 * k2);
            ps->bench.dtoh += time_dtoh / ((double) (n2 * k2));
            ps->bench.htod += time_htod / ((double) (n2 * n2));
        }
    }
    /* use average over all trials */
    ps->bench.gpu /= times;
    ps->bench.cpu /= times;
    ps->bench.cpu *= 1.5;
    ps->bench.dtoh /= times;
    ps->bench.htod /= times;
    ps->bench.times = 1;

    gpaw_cubSCall(cublasSetStream(_gpaw_cublas_handle, 0));
    cudaFree(a_gpu);
    cudaFree(b_gpu);
}


extern "C"
static void hybrid_gemm_update_paces(hybrid_func_params_t *pg)
{
    float time_gpu;
    float time_cpu;
    float time_dtoh;
    float time_htod;

    gpaw_cudaSafeCall(cudaEventSynchronize(pg->event_htod[1]));
    gpaw_cudaSafeCall(cudaEventSynchronize(pg->event_gpu[1]));

    cudaEventElapsedTime(&time_gpu, pg->event_gpu[0], pg->event_gpu[1]);
    cudaEventElapsedTime(&time_cpu, pg->event_dtoh[1], pg->event_htod[0]);
    cudaEventElapsedTime(&time_dtoh, pg->event_dtoh[0], pg->event_dtoh[1]);
    cudaEventElapsedTime(&time_htod, pg->event_htod[0], pg->event_htod[1]);

    time_gpu /= (double) pg->m1 * pg->n1 * pg->k;
    time_cpu /= (double) pg->m2 * pg->n2 * pg->k;
    if (pg->beta_null)
        time_dtoh /= (double) (pg->m2 * pg->k + pg->k * pg->n2);
    else
        time_dtoh /= (double)
                     (pg->m2 * pg->k + pg->k * pg->n2 + pg->n2 * pg->m2);
    time_htod /= (double) (pg->n2 * pg->m2);

    hybrid_pace_t *pace = hybrid_pace_get(pg->pace, HYBRID_MAX_PACE,
                                          pg->m, pg->k, pg->n, pg->ndouble);

    hybrid_pace_update(pace, time_gpu, time_cpu, time_dtoh, time_htod);
    pg->hybrid = 0;
}


extern "C"
static void hybrid_syrk_update_paces(hybrid_func_params_t *ps)
{
    float time_gpu1, time_gpu2;
    float time_cpu;
    float time_dtoh;
    float time_htod;

    gpaw_cudaSafeCall(cudaEventSynchronize(ps->event_gpu[3]));

    cudaEventElapsedTime(&time_gpu1, ps->event_gpu[0], ps->event_gpu[1]);
    cudaEventElapsedTime(&time_gpu2, ps->event_gpu[2], ps->event_gpu[3]);
    cudaEventElapsedTime(&time_cpu, ps->event_dtoh[1], ps->event_htod[0]);
    cudaEventElapsedTime(&time_dtoh, ps->event_dtoh[0], ps->event_dtoh[1]);
    cudaEventElapsedTime(&time_htod, ps->event_htod[0], ps->event_htod[1]);

    time_gpu1 /= (double) ps->n * ps->n * ps->k1;
    time_cpu /= (double) ps->n * ps->n * ps->k2;
    time_dtoh /= (double) (ps->n * ps->k2);
    time_htod /= (double) (ps->n * ps->n);

    hybrid_pace_t *pace = hybrid_pace_get(ps->pace, HYBRID_MAX_PACE,
                                          ps->n, ps->k, ps->n, ps->ndouble);

    hybrid_pace_update(pace, time_gpu1, time_cpu, time_dtoh, time_htod);
    ps->hybrid = 0;
}


extern "C"
static void hybrid_syr2k_update_paces(hybrid_func_params_t *ps)
{
    float time_gpu1, time_gpu2;
    float time_cpu;
    float time_dtoh;
    float time_htod;

    gpaw_cudaSafeCall(cudaEventSynchronize(ps->event_gpu[3]));

    cudaEventElapsedTime(&time_gpu1, ps->event_gpu[0], ps->event_gpu[1]);
    cudaEventElapsedTime(&time_gpu2, ps->event_gpu[2], ps->event_gpu[3]);
    cudaEventElapsedTime(&time_cpu, ps->event_dtoh[1], ps->event_htod[0]);
    cudaEventElapsedTime(&time_dtoh, ps->event_dtoh[0], ps->event_dtoh[1]);
    cudaEventElapsedTime(&time_htod, ps->event_htod[0], ps->event_htod[1]);

    time_gpu1 /= (double) ps->n * ps->n * ps->k1;
    time_cpu /= (double) ps->n * ps->n * ps->k2;
    time_dtoh /= (double) (2 * ps->n * ps->k2);
    time_htod /= (double) (ps->n * ps->n);

    hybrid_pace_t *pace = hybrid_pace_get(ps->pace, HYBRID_MAX_PACE,
                                          ps->n, ps->k, ps->n, ps->ndouble);

    hybrid_pace_update(pace, time_gpu1, time_cpu, time_dtoh, time_htod);
    ps->hybrid = 0;
}


extern "C"
cublasOperation_t cublas_operation(int op)
{
    cublasOperation_t cu_op;

    if (op == 'N' || op == 'n')
        cu_op = CUBLAS_OP_N;
    else if (op == 'T' || op == 't')
        cu_op = CUBLAS_OP_T;
    else if (op == 'C' || op == 'c')
        cu_op = CUBLAS_OP_C;
    else
        assert(0);
    return cu_op;
}


extern "C"
PyObject* scal_cuda_gpu(PyObject *self, PyObject *args)
{
    Py_complex alpha;

    CUdeviceptr x_gpu;
    PyObject *x_shape;
    PyArray_Descr *type;

    if (!PyArg_ParseTuple(args, "DnOO", &alpha, &x_gpu, &x_shape, &type))
        return NULL;

    int n = (int) PyLong_AsLong(PyTuple_GetItem(x_shape, 0));
    Py_ssize_t nd = PyTuple_Size(x_shape);
    for (int d=1; d < nd; d++)
        n *= (int) PyLong_AsLong(PyTuple_GetItem(x_shape, d));
    int incx = 1;
    if (type->type_num == NPY_DOUBLE) {
        gpaw_cubSCall(
                cublasDscal(_gpaw_cublas_handle, n, &alpha.real,
                            (double*) x_gpu, incx));
    } else {
        cuDoubleComplex alpha_gpu = {alpha.real, alpha.imag};
        gpaw_cubSCall(
                cublasZscal(_gpaw_cublas_handle, n, &alpha_gpu,
                    (cuDoubleComplex*) x_gpu, incx));
    }
    if (PyErr_Occurred())
        return NULL;
    else
        Py_RETURN_NONE;
}


extern "C"
PyObject* mmm_gpu(PyObject *self, PyObject *args)
{
    Py_complex alpha;
    CUdeviceptr b;
    int ldb;
    int opb;
    CUdeviceptr a;
    int lda;
    int opa;
    Py_complex beta;
    CUdeviceptr c;
    int ldc;
    int bytes;
    int m, n, k;

    if (!PyArg_ParseTuple(args, "DniCniCDniiiii",
                          &alpha, &b, &ldb, &opb, &a, &lda, &opa,
                          &beta, &c, &ldc, &bytes, &m, &n, &k))
        return NULL;

    cublasOperation_t cu_opa;
    cublasOperation_t cu_opb;

    cu_opa = cublas_operation(opa);
    cu_opb = cublas_operation(opb);

    if (bytes == NPY_SIZEOF_DOUBLE) {
        gpaw_cubSCall(
                cublasDgemm(_gpaw_cublas_handle, cu_opa, cu_opb, m, n, k,
                            &(alpha.real), (double*) a, lda, (double*) b, ldb,
                            &(beta.real), (double*) c, ldc));
    } else {
        cuDoubleComplex cu_alpha = {alpha.real, alpha.imag};
        cuDoubleComplex cu_beta = {beta.real, beta.imag};
        gpaw_cubSCall(
                cublasZgemm(_gpaw_cublas_handle, cu_opa, cu_opb, m, n, k,
                            &cu_alpha, (cuDoubleComplex*) a, lda,
                            (cuDoubleComplex*) b, ldb,
                            &cu_beta, (cuDoubleComplex*) c, ldc));
    }

    Py_RETURN_NONE;
}

static void _gemm_cuda(char transa, cublasOperation_t transa_c,
                       int m, int n, int k,
                       Py_complex alpha, CUdeviceptr a_gpu, int lda,
                       CUdeviceptr b_gpu, int ldb, Py_complex beta,
                       CUdeviceptr c_gpu, int ldc,
                       bool real)
{
    if (real) {
        gpaw_cubSCall(
                cublasDgemm(_gpaw_cublas_handle, transa_c, CUBLAS_OP_N,
                            m, n, k,
                            &alpha.real, (double*) a_gpu, lda,
                            (double*) b_gpu, ldb,
                            &beta.real, (double*) c_gpu, ldc));
    } else {
        cuDoubleComplex alpha_gpu = {alpha.real, alpha.imag};
        cuDoubleComplex beta_gpu = {beta.real, beta.imag};
        gpaw_cubSCall(
                cublasZgemm(_gpaw_cublas_handle, transa_c, CUBLAS_OP_N,
                            m, n, k,
                            &alpha_gpu, (cuDoubleComplex*) a_gpu, lda,
                            (cuDoubleComplex*) b_gpu, ldb,
                            &beta_gpu, (cuDoubleComplex*) c_gpu, ldc));
    }
}

static void _gemm_cuda_hybrid(char transa, cublasOperation_t transa_c,
                              int m, int n, int k,
                              Py_complex alpha, CUdeviceptr a_gpu, int lda,
                              CUdeviceptr b_gpu, int ldb, Py_complex beta,
                              CUdeviceptr c_gpu, int ldc,
                              bool real)
{
    int n_off = 0, m_off = 0;
    int lda2, ldc2;
    int beta_null = 0;

    hybrid_func_params_t *pg = &hybrid_gemm_params;
    hybrid_params_t *ph = &hybrid_params;

    if (!pg->init)
        hybrid_func_init(pg);
    if (!pg->bench.times)
        hybrid_gemm_benchmark(pg, ph);
    if (pg->hybrid)
        hybrid_gemm_update_paces(pg);

    if (beta.real < DBL_MIN && beta.real > -DBL_MIN &&
            beta.imag < DBL_MIN && beta.imag > -DBL_MIN)
        beta_null = 1;

    pg->ndouble = (real) ? 1 : 2;

    hybrid_pace_t *pace = hybrid_pace_get(pg->pace, HYBRID_MAX_PACE,
                                          m, k, n, pg->ndouble);
    hybrid_pace_t *paceu = (pace->times == 0) ? &pg->bench : pace;

    if (beta_null) {
        pg->n2 = (paceu->gpu * m * n * k - paceu->dtoh * k * m)
               / (paceu->gpu * m * k + paceu->cpu * m * k
                  + paceu->dtoh * (k) + paceu->htod * m);
        pg->m2 = (paceu->gpu * m * n * k - paceu->dtoh * k * n)
               / (paceu->gpu * n * k + paceu->cpu * n * k
                  + paceu->dtoh * (k) + paceu->htod * n);
    } else {
        pg->n2 = (paceu->gpu * m * n * k - paceu->dtoh * k * m)
               / (paceu->gpu * m * k + paceu->cpu * m * k
                  + paceu->dtoh * (k + m) + paceu->htod * m);
        pg->m2 = (paceu->gpu * m * n * k - paceu->dtoh * k * n)
               / (paceu->gpu * n * k + paceu->cpu * n * k
                  + paceu->dtoh * (k + n) + paceu->htod * n);
    }
    if (pg->m2 * n > pg->n2 * m) {
        pg->n2 = pg->n1 = n;
        n_off = 0;
        pg->m1 = MIN(m, HYBRID_GEMM_GPU_MDIV
                        * ((m - pg->m2 + HYBRID_GEMM_GPU_MDIV - 1)
                           / HYBRID_GEMM_GPU_MDIV));
        if (pg->m1 == m)
            pg->m1 = MIN(m, HYBRID_GEMM_GPU_NDIV
                            * ((m - pg->m2 + HYBRID_GEMM_GPU_NDIV - 1)
                               / HYBRID_GEMM_GPU_NDIV));
        if (pg->m1 == m)
            pg->m1 = MIN(m, 2 * ((m - pg->m2 + 1) / 2));
        pg->m2 = m - pg->m1;
        m_off = pg->m1;
    } else {
        pg->n1 = MIN(n, HYBRID_GEMM_GPU_NDIV
                        * ((n - pg->n2 + HYBRID_GEMM_GPU_NDIV - 1)
                           / HYBRID_GEMM_GPU_NDIV));
        if (pg->n1 == n)
            pg->n1 = MIN(n, 2 * ((n - pg->n2 + 1) / 2));
        pg->n2 = n - pg->n1;
        n_off = pg->n1;
        pg->m1 = pg->m2 = m;
        m_off = 0;
    }
    if (pg->n2 > 1 && pg->n2 <= n && pg->m2 > 1 && pg->m2 <= m) {
        if (transa == 'n')
            lda2 = pg->m2;
        else
            lda2 = lda;
        ldc2 = pg->m2;
        pg->k = k;
        pg->m = m;
        pg->n = n;
        pg->beta_null = beta_null;
        pace->times = MIN(HYBRID_FUNC_MAX_TIMES, pace->times + 1);
        hybrid_param_alloc(ph, k * pg->m2 * pg->ndouble,
                           k * pg->n2 * pg->ndouble,
                           pg->n2 * pg->m2 * pg->ndouble, 0);
        pg->hybrid = 1;
    } else {
        pg->hybrid = 0;
        _gemm_cuda(transa, transa_c, m, n, k,
                   alpha, a_gpu, lda, b_gpu, ldb, beta, c_gpu, ldc, real);
        return;
    }

    gpaw_cubSCall(
            cublasSetStream(_gpaw_cublas_handle, ph->stream[0]));
    cudaEventRecord(pg->event_gpu[0], ph->stream[0]);
    if (real) {
        gpaw_cubSCall(
                cublasDgemm(_gpaw_cublas_handle, transa_c, CUBLAS_OP_N,
                            pg->m1, pg->n1, k,
                            &alpha.real, (double*) a_gpu, lda,
                            (double*) b_gpu, ldb,
                            &beta.real, (double*) c_gpu, ldc));
    } else {
        cuDoubleComplex alpha_gpu = {alpha.real, alpha.imag};
        cuDoubleComplex beta_gpu = {beta.real, beta.imag};
        gpaw_cubSCall(
                cublasZgemm(_gpaw_cublas_handle, transa_c, CUBLAS_OP_N,
                            pg->m1, pg->n1, k,
                            &alpha_gpu, (cuDoubleComplex*) a_gpu, lda,
                            (cuDoubleComplex*) b_gpu, ldb,
                            &beta_gpu, (cuDoubleComplex*) c_gpu, ldc));
    }
    cudaEventRecord(pg->event_gpu[1], ph->stream[0]);
    gpaw_cubSCall(
            cublasSetStream(_gpaw_cublas_handle, 0));
    cudaEventRecord(pg->event_dtoh[0], ph->stream[1]);
    if (transa == 'n') {
        gpaw_cubSCall(
                cublasGetMatrixAsync(pg->m2, k,
                                     sizeof(double) * pg->ndouble,
                                     (void*) ((double*) a_gpu
                                         + m_off * pg->ndouble),
                                     lda, (void*) ph->a, lda2,
                                     ph->stream[1]));
    } else {
        gpaw_cubSCall(
                cublasGetMatrixAsync(k, pg->m2,
                                     sizeof(double) * pg->ndouble,
                                     (void*) ((double*) a_gpu
                                         + lda * m_off * pg->ndouble),
                                     lda, (void*) ph->a, lda2,
                                     ph->stream[1]));
    }
    gpaw_cubSCall(
            cublasGetMatrixAsync(k, pg->n2,
                                 sizeof(double) * pg->ndouble,
                                 (void*) ((double*) b_gpu
                                     + ldb * n_off * pg->ndouble),
                                 ldb, (void*) ph->b, ldb,
                                 ph->stream[1]));
    if (!beta_null) {
        gpaw_cubSCall(
                cublasGetMatrixAsync(pg->m2, pg->n2,
                                     sizeof(double) * pg->ndouble,
                                     (void*) ((double*) c_gpu
                                         + n_off * ldc * pg->ndouble
                                         + m_off * pg->ndouble),
                                     ldc, (void*) ph->c, ldc2,
                                     ph->stream[1]));
    }
    cudaEventRecord(pg->event_dtoh[1], ph->stream[1]);
    Py_BEGIN_ALLOW_THREADS;
    gpaw_cudaSafeCall(cudaEventSynchronize(pg->event_dtoh[1]));
    if (real) {
        dgemm_(&transa, "n", &pg->m2, &pg->n2, &k,
               &(alpha.real), ph->a, &lda2, ph->b, &ldb,
               &(beta.real), ph->c, &ldc2);
    } else {
        zgemm_(&transa, "n", &pg->m2, &pg->n2, &k,
               &alpha, (void*) ph->a, &lda2, (void*) ph->b, &ldb,
               &beta, (void*) ph->c, &ldc2);
    }
    Py_END_ALLOW_THREADS;
    cudaEventRecord(pg->event_htod[0], ph->stream[1]);
    gpaw_cubSCall(
            cublasSetMatrixAsync(pg->m2, pg->n2,
                                 sizeof(double) * pg->ndouble,
                                 (void*) ph->c, ldc2,
                                 (void*) ((double*) c_gpu
                                     + n_off * ldc * pg->ndouble
                                     + m_off * pg->ndouble),
                                 ldc, ph->stream[1]));
    cudaEventRecord(pg->event_htod[1], ph->stream[1]);
    cudaStreamWaitEvent(0, pg->event_htod[1], 0);
    cudaStreamWaitEvent(0, pg->event_gpu[1], 0);
}

extern "C"
PyObject* gemm_cuda_gpu(PyObject *self, PyObject *args)
{
    Py_complex alpha;
    Py_complex beta;

    CUdeviceptr a_gpu;
    CUdeviceptr b_gpu;
    CUdeviceptr c_gpu;
    PyObject *a_shape, *b_shape, *c_shape;
    PyArray_Descr *type;

    char transa = 'n';
    int  hybrid = 0;

    if (!PyArg_ParseTuple(args, "DnOnODnOO|Ci", &alpha, &a_gpu, &a_shape,
                          &b_gpu, &b_shape, &beta, &c_gpu, &c_shape, &type,
                          &transa, &hybrid))
        return NULL;

    bool real = 0;
    if (type->type_num == NPY_DOUBLE) {
        real = 1;
    }

    cublasOperation_t transa_c = cublas_operation(transa);

    int m, k, lda, ldb, ldc;
    int n = (int) PyLong_AsLong(PyTuple_GetItem(b_shape, 0));
    if (transa == 'n') {
        m = (int) PyLong_AsLong(PyTuple_GetItem(a_shape, 1));
        for (int i=2; i < PyTuple_Size(a_shape); i++)
            m *= (int) PyLong_AsLong(PyTuple_GetItem(a_shape, i));
        k = (int) PyLong_AsLong(PyTuple_GetItem(a_shape, 0));
        lda = m;
        ldb = k;
        ldc = m;
    } else {
        k = (int) PyLong_AsLong(PyTuple_GetItem(a_shape, 1));
        for (int i=2; i < PyTuple_Size(a_shape); i++)
            k *= (int) PyLong_AsLong(PyTuple_GetItem(a_shape, i));
        m = (int) PyLong_AsLong(PyTuple_GetItem(a_shape, 0));
        lda = k;
        ldb = k;
        ldc = m;
    }

    if (hybrid) {
        _gemm_cuda_hybrid(transa, transa_c, m, n, k,
                          alpha, a_gpu, lda, b_gpu, ldb, beta,
                          c_gpu, ldc, real);
    } else {
        _gemm_cuda(transa, transa_c, m, n, k,
                   alpha, a_gpu, lda, b_gpu, ldb, beta,
                   c_gpu, ldc, real);
    }

    if (PyErr_Occurred())
        return NULL;
    else
        Py_RETURN_NONE;
}


extern "C"
PyObject* gemv_cuda_gpu(PyObject *self, PyObject *args)
{
    Py_complex alpha;

    CUdeviceptr a_gpu;
    CUdeviceptr x_gpu;
    CUdeviceptr y_gpu;

    Py_complex beta;
    PyObject *a_shape, *x_shape;
    PyArray_Descr *type;

    int trans = 't';
    if (!PyArg_ParseTuple(args, "DnOnODnO|C", &alpha, &a_gpu, &a_shape,
                          &x_gpu, &x_shape, &beta, &y_gpu, &type, &trans))
        return NULL;

    cublasOperation_t trans_c = cublas_operation(trans);

    int m, n, lda, incx, incy;
    if (trans == 'n') {
        m = (int) PyLong_AsLong(PyTuple_GetItem(a_shape, 1));
        for (int i=2; i < PyTuple_Size(a_shape); i++)
            m *= (int) PyLong_AsLong(PyTuple_GetItem(a_shape, i));
        n = (int) PyLong_AsLong(PyTuple_GetItem(a_shape, 0));
        lda = m;
    } else {
        n = (int) PyLong_AsLong(PyTuple_GetItem(a_shape, 0));
        for (int i=1; i < PyTuple_Size(a_shape) - 1; i++)
            n *= (int) PyLong_AsLong(PyTuple_GetItem(a_shape, i));
        m = (int) PyLong_AsLong(
                PyTuple_GetItem(a_shape, PyTuple_Size(a_shape) - 1));
        lda = m;
    }

    incx = 1;
    incy = 1;
    if (type->type_num == NPY_DOUBLE) {
        gpaw_cubSCall(
                cublasDgemv(_gpaw_cublas_handle, trans_c, m, n,
                            &alpha.real, (double*) a_gpu, lda,
                            (double*) x_gpu, incx,
                            &beta.real, (double*) y_gpu, incy));
    } else {
        cuDoubleComplex alpha_gpu = {alpha.real, alpha.imag};
        cuDoubleComplex beta_gpu = {beta.real, beta.imag};
        gpaw_cubSCall(
                cublasZgemv(_gpaw_cublas_handle, trans_c, m, n,
                            &alpha_gpu, (cuDoubleComplex*) a_gpu, lda,
                            (cuDoubleComplex*) x_gpu, incx,
                            &beta_gpu, (cuDoubleComplex*) y_gpu, incy));
    }
    if (PyErr_Occurred())
        return NULL;
    else
        Py_RETURN_NONE;
}


extern "C"
PyObject* axpy_cuda_gpu(PyObject *self, PyObject *args)
{
    Py_complex alpha;

    CUdeviceptr x_gpu;
    CUdeviceptr y_gpu;
    PyObject *x_shape,*y_shape;
    PyArray_Descr *type;

    if (!PyArg_ParseTuple(args, "DnOnOO", &alpha, &x_gpu, &x_shape,
                          &y_gpu, &y_shape, &type))
        return NULL;

    Py_ssize_t nd = PyTuple_Size(x_shape);
    int n = (int) PyLong_AsLong(PyTuple_GetItem(x_shape, 0));
    for (int d=1; d < nd; d++)
        n *= (int) PyLong_AsLong(PyTuple_GetItem(x_shape, d));
    int incx = 1;
    int incy = 1;
    if (type->type_num == NPY_DOUBLE) {
        gpaw_cubSCall(
                cublasDaxpy(_gpaw_cublas_handle, n, &alpha.real,
                            (double*) x_gpu, incx,
                            (double*) y_gpu, incy));
    } else {
        cuDoubleComplex alpha_gpu = {alpha.real, alpha.imag};
        gpaw_cubSCall(
                cublasZaxpy(_gpaw_cublas_handle, n, &alpha_gpu,
                            (cuDoubleComplex*) x_gpu, incx,
                            (cuDoubleComplex*) y_gpu, incy));
    }
    if (PyErr_Occurred())
        return NULL;
    else
        Py_RETURN_NONE;
}

static void _rk_cuda_gpu(int n, int k,
                         double alpha, CUdeviceptr a_gpu, int lda,
                         double beta, CUdeviceptr c_gpu, int ldc,
                         bool real)
{
    if (real) {
        gpaw_cubSCall(
                cublasDsyrk(_gpaw_cublas_handle,
                    CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T,
                    n, k,
                    &alpha, (double*) a_gpu, lda,
                    &beta, (double*) c_gpu, ldc));
    } else {
        gpaw_cubSCall(
                cublasZherk(_gpaw_cublas_handle,
                    CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T,
                    n, k,
                    &alpha, (cuDoubleComplex*) a_gpu, lda,
                    &beta, (cuDoubleComplex*) c_gpu, ldc));
    }
}

static void _rk_cuda_gpu_hybrid(int n, int k,
                                double alpha, CUdeviceptr a_gpu, int lda,
                                double beta, CUdeviceptr c_gpu, int ldc,
                                bool real)
{
    double beta2=0;
    int lda2;

    hybrid_func_params_t *ps = &hybrid_syrk_params;
    hybrid_params_t *ph = &hybrid_params;

    if (!ps->init)
        hybrid_func_init(ps);
    if (!ps->bench.times)
        hybrid_syrk_benchmark(ps, ph);
    if (ps->hybrid)
        hybrid_syrk_update_paces(ps);

    ps->ndouble = (real) ? 1 : 2;

    hybrid_pace_t *pace = hybrid_pace_get(ps->pace, HYBRID_MAX_PACE,
                                          n, k, n, ps->ndouble);
    hybrid_pace_t *paceu = (pace->times == 0) ? &ps->bench : pace;

    ps->k2 = n * (paceu->gpu * k - paceu->htod)
           / (paceu->cpu * n + paceu->gpu * n + paceu->dtoh);
    ps->k1 = MIN(k, HYBRID_SYRK_GPU_KDIV
                    * ((k - ps->k2 + HYBRID_SYRK_GPU_KDIV - 1)
                       / HYBRID_SYRK_GPU_KDIV));
    if (ps->k1 == k)
        ps->k1 = MIN(k, 2 * ((k - ps->k2 + 1) / 2));

    ps->k2 = k - ps->k1;
    if (ps->k2 > 1 && ps->k2 <= k) {
        pace->times = MIN(HYBRID_FUNC_MAX_TIMES, pace->times + 1);
        hybrid_param_alloc(ph, ps->k2 * n * ps->ndouble, 0,
                           n * n * ps->ndouble, n * n * ps->ndouble);
        ps->k = k;
        ps->n = n;
        lda2 = ps->k2;
        ps->hybrid = 1;
    } else {
        ps->hybrid = 0;
        _rk_cuda_gpu(n, k, alpha, a_gpu, lda, beta, c_gpu, ldc, real);
        return;
    }

    gpaw_cubSCall(cublasSetStream(_gpaw_cublas_handle, ph->stream[0]));
    cudaEventRecord(ps->event_gpu[0], ph->stream[0]);
    if (real) {
        gpaw_cubSCall(
                cublasDsyrk(_gpaw_cublas_handle,
                    CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T,
                    n, ps->k1,
                    &alpha, (double*) a_gpu, lda,
                    &beta, (double*) c_gpu, ldc));
    } else {
        gpaw_cubSCall(
                cublasZherk(_gpaw_cublas_handle,
                    CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T,
                    n, ps->k1,
                    &alpha, (cuDoubleComplex*) a_gpu, lda,
                    &beta, (cuDoubleComplex*) c_gpu, ldc));
    }
    cudaEventRecord(ps->event_gpu[1], ph->stream[0]);
    cudaEventRecord(ps->event_gpu[3], ph->stream[2]);
    gpaw_cubSCall(cublasSetStream(_gpaw_cublas_handle, 0));

    cudaEventRecord(ps->event_dtoh[0], ph->stream[1]);
    gpaw_cubSCall(
            cublasGetMatrixAsync(ps->k2, n,
                                 sizeof(double) * ps->ndouble,
                                 (void*) ((double*) a_gpu
                                     + ps->k1 * ps->ndouble),
                                 lda, (void*) ph->a, lda2,
                                 ph->stream[1]));
    memset(ph->c, 0, sizeof(double) * ps->ndouble * ps->n * ps->n);
    cudaEventRecord(ps->event_dtoh[1], ph->stream[1]);

    Py_BEGIN_ALLOW_THREADS;
    gpaw_cudaSafeCall(cudaEventSynchronize(ps->event_dtoh[1]));
    if (real) {
        dsyrk_("u", "t", &ps->n, &ps->k2,
               &alpha, ph->a, &lda2,
               &beta2, ph->c, &ldc);
    } else {
        zherk_("u", "c", &ps->n, &ps->k2,
               &alpha, (void*) ph->a, &lda2,
               &beta2, (void*) ph->c, &ldc);
    }
    Py_END_ALLOW_THREADS;

    cudaEventRecord(ps->event_htod[0], ph->stream[1]);
    gpaw_cubSCall(
            cublasSetMatrixAsync(ps->n, ps->n,
                                 sizeof(double) * ps->ndouble,
                                 (void*) ph->c, ldc,
                                 (void*) ((double*) ph->c_gpu),
                                 ldc, ph->stream[1]));
    cudaEventRecord(ps->event_htod[1], ph->stream[1]);

    double alpha_=1;
    cudaStreamWaitEvent(0, ps->event_gpu[1], 0);
    cudaStreamWaitEvent(0, ps->event_htod[1], 0);
    cudaEventRecord(ps->event_gpu[2], 0);
    gpaw_cubSCall(
            cublasDaxpy(_gpaw_cublas_handle, ps->n * ps->n * ps->ndouble,
                &alpha_, (double*) ph->c_gpu, 1, (double*) c_gpu, 1));
    cudaEventRecord(ps->event_gpu[3], 0);
}

extern "C"
PyObject* rk_cuda_gpu(PyObject *self, PyObject *args)
{
    double alpha;
    double beta;

    CUdeviceptr a_gpu;
    CUdeviceptr c_gpu;
    PyObject *a_shape, *c_shape;
    PyArray_Descr *type;
    int hybrid = 0;

    if (!PyArg_ParseTuple(args, "dnOdnOO|i", &alpha, &a_gpu, &a_shape,
                          &beta, &c_gpu, &c_shape, &type, &hybrid))
        return NULL;

    bool real = 0;
    if (type->type_num == NPY_DOUBLE) {
        real = 1;
    }

    int n = (int) PyLong_AsLong(PyTuple_GetItem(a_shape, 0));
    int k = (int) PyLong_AsLong(PyTuple_GetItem(a_shape, 1));
    for (int d=2; d < PyTuple_Size(a_shape); d++)
        k *= (int) PyLong_AsLong(PyTuple_GetItem(a_shape, d));
    int ldc = n;
    int lda = k;

    if (hybrid) {
        _rk_cuda_gpu_hybrid(n, k, alpha, a_gpu, lda, beta, c_gpu, ldc, real);
    } else {
        _rk_cuda_gpu(n, k, alpha, a_gpu, lda, beta, c_gpu, ldc, real);
    }

    if (PyErr_Occurred())
        return NULL;
    else
        Py_RETURN_NONE;
}

static void _r2k_cuda_gpu(int n, int k,
                          Py_complex alpha, CUdeviceptr a_gpu, int lda,
                          CUdeviceptr b_gpu, double beta,
                          CUdeviceptr c_gpu, int ldc, bool real)
{
    if (real) {
        gpaw_cubSCall(
                cublasDsyr2k(_gpaw_cublas_handle,
                    CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, n, k,
                    &alpha.real, (double*) a_gpu, lda,
                    (double*) b_gpu, lda,
                    &beta, (double*) c_gpu, ldc));
    } else {
        cuDoubleComplex alpha_gpu = {alpha.real, alpha.imag};
        gpaw_cubSCall(
                cublasZher2k(_gpaw_cublas_handle,
                    CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, n, k,
                    &alpha_gpu, (cuDoubleComplex*) a_gpu, lda,
                    (cuDoubleComplex*) b_gpu, lda,
                    &beta, (cuDoubleComplex*) c_gpu, ldc));
    }
}

static void _r2k_cuda_gpu_hybrid(int n, int k,
                                 Py_complex alpha, CUdeviceptr a_gpu, int lda,
                                 CUdeviceptr b_gpu, double beta,
                                 CUdeviceptr c_gpu, int ldc, bool real)
{
    double beta2 = 0;
    int lda2;

    hybrid_func_params_t *ps2 = &hybrid_syr2k_params;
    hybrid_params_t *ph = &hybrid_params;

    if (!ps2->init)
        hybrid_func_init(ps2);
    if (!ps2->bench.times)
        hybrid_syr2k_benchmark(ps2, ph);
    if (ps2->hybrid)
        hybrid_syr2k_update_paces(ps2);

    ps2->ndouble = (real) ? 1 : 2;

    hybrid_pace_t *pace = hybrid_pace_get(ps2->pace, HYBRID_MAX_PACE,
                                          n, k, n, ps2->ndouble);
    hybrid_pace_t *paceu = (pace->times == 0) ? &ps2->bench : pace;

    ps2->k2 = n * (paceu->gpu * k - paceu->htod)
            / (paceu->cpu * n + paceu->gpu * n + 2 * paceu->dtoh);

    ps2->k1 = MIN(k, HYBRID_SYR2K_GPU_KDIV
                     * ((k - ps2->k2 + HYBRID_SYR2K_GPU_KDIV - 1)
                        / HYBRID_SYR2K_GPU_KDIV));
    if (ps2->k1 == k)
        ps2->k1 = MIN(k, 2 * ((k - ps2->k2 + 1) / 2));

    ps2->k2 = k - ps2->k1;
    if (ps2->k2 > 1 && ps2->k2 <= k) {
        pace->times = MIN(HYBRID_FUNC_MAX_TIMES, pace->times + 1);
        hybrid_param_alloc(ph, ps2->k2 * n * ps2->ndouble,
                           ps2->k2 * n * ps2->ndouble,
                           n * n * ps2->ndouble, n * n * ps2->ndouble);
        ps2->k = k;
        ps2->n = n;
        lda2 = ps2->k2;
        ps2->hybrid = 1;
    } else {
        ps2->hybrid = 0;
        _r2k_cuda_gpu(n, k, alpha, a_gpu, lda, b_gpu, beta,
                      c_gpu, ldc, real);
        return;
    }

    gpaw_cubSCall(
            cublasSetStream(_gpaw_cublas_handle, ph->stream[0]));
    cudaEventRecord(ps2->event_gpu[0], ph->stream[0]);

    if (real) {
        gpaw_cubSCall(
                cublasDsyr2k(_gpaw_cublas_handle,
                    CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, n, ps2->k1,
                    &alpha.real, (double*) a_gpu, lda,
                    (double*) b_gpu, lda,
                    &beta, (double*) c_gpu, ldc));
    } else {
        cuDoubleComplex alpha_gpu = {alpha.real, alpha.imag};
        gpaw_cubSCall(
                cublasZher2k(_gpaw_cublas_handle,
                    CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, n, ps2->k1,
                    &alpha_gpu, (cuDoubleComplex*) a_gpu, lda,
                    (cuDoubleComplex*) b_gpu, lda,
                    &beta, (cuDoubleComplex*) c_gpu, ldc));
    }
    cudaEventRecord(ps2->event_gpu[1], ph->stream[0]);
    cudaEventRecord(ps2->event_gpu[3], ph->stream[2]);
    gpaw_cubSCall(cublasSetStream(_gpaw_cublas_handle, 0));

    cudaEventRecord(ps2->event_dtoh[0], ph->stream[1]);
    gpaw_cubSCall(
            cublasGetMatrixAsync(ps2->k2, n,
                                 sizeof(double) * ps2->ndouble,
                                 (void*) ((double*) a_gpu
                                     + ps2->k1 * ps2->ndouble),
                                 lda, (void*) ph->a, lda2,
                                 ph->stream[1]));
    gpaw_cubSCall(
            cublasGetMatrixAsync(ps2->k2, n,
                                 sizeof(double) * ps2->ndouble,
                                 (void*) ((double*) b_gpu
                                     + ps2->k1 * ps2->ndouble),
                                 lda, (void*) ph->b, lda2,
                                 ph->stream[1]));
    memset(ph->c, 0, sizeof(double) * ps2->ndouble * ps2->n * ps2->n);
    cudaEventRecord(ps2->event_dtoh[1], ph->stream[1]);

    Py_BEGIN_ALLOW_THREADS;
    gpaw_cudaSafeCall(cudaEventSynchronize(ps2->event_dtoh[1]));
    if (real) {
        dsyr2k_("u", "t", &ps2->n, &ps2->k2,
                &alpha.real, ph->a, &lda2, ph->b, &lda2, &beta2,
                ph->c, &ldc);
    } else {
        zher2k_("u", "c", &ps2->n, &ps2->k2,
                &alpha, ph->a, &lda2,  ph->b, &lda2, &beta2,
                (void*) ph->c, &ldc);
    }
    Py_END_ALLOW_THREADS;

    cudaEventRecord(ps2->event_htod[0],ph->stream[1]);
    gpaw_cubSCall(
            cublasSetMatrixAsync(ps2->n, ps2->n,
                                 sizeof(double) * ps2->ndouble,
                                 (void*) ph->c, ldc,
                                 (void*) ((double*) ph->c_gpu),
                                 ldc, ph->stream[1]));
    cudaEventRecord(ps2->event_htod[1], ph->stream[1]);

    double alpha_ = 1;
    cudaStreamWaitEvent(0, ps2->event_gpu[1], 0);
    cudaStreamWaitEvent(0, ps2->event_htod[1], 0);
    cudaEventRecord(ps2->event_gpu[2], 0);
    gpaw_cubSCall(
            cublasDaxpy(_gpaw_cublas_handle,
                        ps2->n * ps2->n * ps2->ndouble,
                        &alpha_, (double*) ph->c_gpu, 1,
                        (double*) c_gpu, 1));
    cudaEventRecord(ps2->event_gpu[3], 0);
}

extern "C"
PyObject* r2k_cuda_gpu(PyObject *self, PyObject *args)
{
    Py_complex alpha;
    double beta;

    CUdeviceptr a_gpu;
    CUdeviceptr b_gpu;
    CUdeviceptr c_gpu;
    PyObject *a_shape, *b_shape, *c_shape;
    PyArray_Descr *type;

    int hybrid = 0;

    if (!PyArg_ParseTuple(args, "DnOnOdnOO|i", &alpha, &a_gpu, &a_shape,
                          &b_gpu, &b_shape, &beta, &c_gpu, &c_shape,
                          &type, &hybrid))
        return NULL;

    bool real = 0;
    if (type->type_num == NPY_DOUBLE) {
        real = 1;
    }

    int n = (int) PyLong_AsLong(PyTuple_GetItem(a_shape, 0));
    int k = (int) PyLong_AsLong(PyTuple_GetItem(a_shape, 1));
    for (int d=2; d < PyTuple_Size(a_shape); d++)
        k *= (int) PyLong_AsLong(PyTuple_GetItem(a_shape, d));
    int ldc = n;
    int lda = k;

    if (hybrid) {
        _r2k_cuda_gpu_hybrid(n, k, alpha, a_gpu, lda, b_gpu, beta,
                             c_gpu, ldc, real);
    } else {
        _r2k_cuda_gpu(n, k, alpha, a_gpu, lda, b_gpu, beta,
                      c_gpu, ldc, real);
    }

    if (PyErr_Occurred())
        return NULL;
    else
        Py_RETURN_NONE;
}


extern "C"
PyObject* dotc_cuda_gpu(PyObject *self, PyObject *args)
{
    CUdeviceptr a_gpu;
    CUdeviceptr b_gpu;

    PyObject *a_shape;
    PyArray_Descr *type;

    if (!PyArg_ParseTuple(args, "nOnO", &a_gpu, &a_shape, &b_gpu, &type))
        return NULL;

    int n = (int) PyLong_AsLong(PyTuple_GetItem(a_shape, 0));
    for (int i=1; i < PyTuple_Size(a_shape); i++)
        n *= (int) PyLong_AsLong(PyTuple_GetItem(a_shape, i));

    int incx = 1;
    int incy = 1;
    if (type->type_num == NPY_DOUBLE) {
        double result;
        gpaw_cubSCall(
                cublasDdot(_gpaw_cublas_handle, n,
                           (double*) a_gpu, incx,
                           (double*) b_gpu, incy, &result));
        if (PyErr_Occurred())
            return NULL;
        else
            return PyFloat_FromDouble(result);
    } else {
        cuDoubleComplex result;
        gpaw_cubSCall(
                cublasZdotc(_gpaw_cublas_handle, n,
                            (cuDoubleComplex*) a_gpu, incx,
                            (cuDoubleComplex*) b_gpu, incy, &result));
        if (PyErr_Occurred())
            return NULL;
        else
            return PyComplex_FromDoubles(result.x,result.y);
    }
}


extern "C"
PyObject* dotu_cuda_gpu(PyObject *self, PyObject *args)
{
    CUdeviceptr a_gpu;
    CUdeviceptr b_gpu;

    PyObject *a_shape;
    PyArray_Descr *type;

    if (!PyArg_ParseTuple(args, "nOnO", &a_gpu, &a_shape, &b_gpu, &type))
        return NULL;

    int n = (int) PyLong_AsLong(PyTuple_GetItem(a_shape, 0));
    for (int i=1; i < PyTuple_Size(a_shape); i++)
        n *= (int) PyLong_AsLong(PyTuple_GetItem(a_shape, i));

    int incx = 1;
    int incy = 1;
    if (type->type_num == NPY_DOUBLE) {
        double result;
        gpaw_cubSCall(
                cublasDdot(_gpaw_cublas_handle, n,
                    (double*) a_gpu, incx,
                    (double*) b_gpu, incy, &result));
        if (PyErr_Occurred())
            return NULL;
        else
            return PyFloat_FromDouble(result);
    } else {
        cuDoubleComplex result;
        gpaw_cubSCall(
                cublasZdotu(_gpaw_cublas_handle, n,
                    (cuDoubleComplex*) a_gpu, incx,
                    (cuDoubleComplex*) b_gpu, incy, &result));
        if (PyErr_Occurred())
            return NULL;
        else
            return PyComplex_FromDoubles(result.x,result.y);
    }
}
