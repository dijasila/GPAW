#include <stdio.h>
#include <malloc.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h>
#include <sys/time.h>

#include "cuda.h"
#include "cuda_runtime_api.h"
#include "bmgs.h"

int main(void)
{
    bmgsstencil s;
    double h[3] = {1.0, 1.0, 1.0};
    long   n[3] = {72,76,76};
    double *a, *b, *b_cuda, error = 0;
    size_t asize, bsize;

    struct timeval t0, t1, t2;
    double flops1, flops2, flops3;

    s = bmgs_laplace(7, 1.0, h, n);

    asize = s.j[0] + s.n[0] * (s.j[1] + s.n[1] * (s.n[2] + s.j[2]));
    bsize = s.n[0] * s.n[1] * s.n[2];

    cudaMallocHost((void**) &a, sizeof(*a) * asize);
    cudaMallocHost((void**) &b, sizeof(*b) * bsize);
    cudaMallocHost((void**) &b_cuda, sizeof(*b_cuda) * bsize);

    fprintf(stdout, "%ld\t", s.ncoefs);
    for (int i=0; i < s.ncoefs; ++i)
        fprintf(stdout, "(%lf %ld)\t", s.coefs[i], s.offsets[i]);
    fprintf(stdout, "\n%ld %ld %ld %ld %ld %ld\n", s.j[0], s.j[1], s.j[2],
            s.n[0], s.n[1], s.n[2]);

    srand((unsigned int) time(NULL));
    for (int i=0; i < asize; i++) {
        a[i] = rand() / (double) RAND_MAX;
    }
    bmgs_fd_cuda(&s, a, b_cuda);

    gettimeofday(&t0, NULL);
    bmgs_fd(&s, a, b);
    gettimeofday(&t1, NULL);
    flops3 = bmgs_fd_cuda(&s, a, b_cuda);
    gettimeofday(&t2, NULL);

    flops1 = t1.tv_sec + t1.tv_usec / 1000000.0 - t0.tv_sec
           - t0.tv_usec / 1000000.0;
    flops2 = t2.tv_sec + t2.tv_usec / 1000000.0 - t1.tv_sec
           - t1.tv_usec / 1000000.0;
    printf("%dx%dx%d  \t CPU: %f\t GPU: %f\t GPU NOMEMTR: %f ms\n", s.n[0],
           s.n[1], s.n[2], 1000 * flops1, 1000 * flops2, 1000 * flops3);

    for (int i=0; i < bsize; i++) {
        error += (b[i] - b_cuda[i]) * (b[i] - b_cuda[i]);
    }
    error = error / (double) bsize;
    fprintf(stdout, "mean sqr error %lf\n", error);

    cudaFree(a);
    cudaFree(b);
    cudaFree(b_cuda);
    return 0;
}
