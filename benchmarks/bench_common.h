/**
 * @file bench_common.h
 * @brief Common utilities for fastpf benchmarks
 */

#ifndef BENCH_COMMON_H
#define BENCH_COMMON_H

#include "fastpf.h"
#include "../src/fastpf_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#ifdef FASTPF_USE_OPENMP
    #include <omp.h>
    #define GET_TIME() omp_get_wtime()
#else
    #define GET_TIME() ((double)clock() / CLOCKS_PER_SEC)
#endif

/* Simple 1D random walk state */
typedef struct {
    double x;
} state_t;

/* Model parameters */
typedef struct {
    double process_std;
    double obs_std;
} model_params_t;

/* Model functions */
void prior_sample(void* ctx, void* x0_out, fastpf_rng_t* rng);
void transition_sample(void* ctx, const void* x_prev, void* x_out, fastpf_rng_t* rng);
double log_likelihood(void* ctx, const void* x, const void* y);

/* Data generation */
void generate_data(double* x_true, double* observations, int T, 
                   double process_std, double obs_std, uint64_t seed);

#endif /* BENCH_COMMON_H */
