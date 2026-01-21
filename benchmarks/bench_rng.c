/**
 * @file bench_rng.c
 * @brief Benchmark pure RNG performance
 */

#include "bench_common.h"
#include <time.h>

#ifdef FASTPF_USE_OPENMP
    #include <omp.h>
    #define GET_TIME() omp_get_wtime()
#else
    #define GET_TIME() ((double)clock() / CLOCKS_PER_SEC)
#endif

int main(int argc, char** argv) {
    fastpf_rng_t rng;
    double t_start, t_end;
    double sum = 0.0;
    int n = 10000000;
    int i;
    
    if (argc > 1) {
        n = atoi(argv[1]);
    }
    
    fastpf_rng_seed(&rng, 42);
    
    printf("Benchmarking fastpf_rng_normal() with %d samples\n", n);
    
    t_start = GET_TIME();
    for (i = 0; i < n; i++) {
        sum += fastpf_rng_normal(&rng);
    }
    t_end = GET_TIME();
    
    printf("Time: %.6f seconds\n", t_end - t_start);
    printf("Rate: %.2f million samples/sec\n", n / (t_end - t_start) / 1e6);
    printf("Per sample: %.2f ns\n", (t_end - t_start) * 1e9 / n);
    printf("Sum: %.6f (sanity check, should be near 0)\n", sum / n);
    
    return 0;
}
