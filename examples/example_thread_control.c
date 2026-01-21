/**
 * @file example_thread_control.c
 * @brief Demonstrates OpenMP thread count control
 */

#include "fastpf.h"
#include "../src/fastpf_internal.h"
#include <stdio.h>
#include <stdlib.h>

typedef struct { double x; } state_t;
typedef struct { double process_std; double obs_std; } params_t;

void prior_sample(void* ctx, void* x0_out, fastpf_rng_t* rng) {
    state_t* s = (state_t*)x0_out;
    (void)ctx;
    s->x = fastpf_rng_normal(rng);
}

void transition_sample(void* ctx, const void* x_prev, void* x_out, fastpf_rng_t* rng) {
    const params_t* p = (const params_t*)ctx;
    const state_t* s_prev = (const state_t*)x_prev;
    state_t* s = (state_t*)x_out;
    s->x = s_prev->x + p->process_std * fastpf_rng_normal(rng);
}

double log_likelihood(void* ctx, const void* x, const void* y) {
    const params_t* p = (const params_t*)ctx;
    const state_t* s = (const state_t*)x;
    const double* obs = (const double*)y;
    double diff = s->x - *obs;
    double var = p->obs_std * p->obs_std;
    return -0.5 * (diff * diff / var);
}

static void run_filter(int N, int num_threads, const char* label) {
    fastpf_cfg_t cfg;
    fastpf_model_t model;
    fastpf_t pf;
    params_t params = {0.1, 0.5};
    double obs = 0.0;
    int k;
    
    fastpf_cfg_init(&cfg, N, sizeof(state_t));
    cfg.rng_seed = 42;
    cfg.num_threads = num_threads;
    
    model.ctx = &params;
    model.prior_sample = prior_sample;
    model.transition_sample = transition_sample;
    model.log_likelihood = log_likelihood;
    model.rejuvenate = NULL;
    
    if (fastpf_init(&pf, &cfg, &model) != FASTPF_SUCCESS) {
        fprintf(stderr, "Init failed\n");
        return;
    }
    
    for (k = 0; k < 100; k++) {
        fastpf_step(&pf, &obs);
    }
    
    printf("%s: Completed 100 steps with N=%d particles\n", label, N);
    
    fastpf_free(&pf);
}

int main(void) {
    printf("Thread Count Control Examples\n");
    printf("==============================\n\n");
    
    printf("Method 1: Environment variable (OMP_NUM_THREADS)\n");
    printf("  export OMP_NUM_THREADS=4\n");
    printf("  ./example_thread_control\n\n");
    
    printf("Method 2: API configuration (cfg.num_threads)\n\n");
    
#ifdef FASTPF_USE_OPENMP
    printf("OpenMP is enabled. Testing different thread counts:\n\n");
    
    run_filter(10000, 0, "Auto (0)");
    run_filter(10000, 1, "Serial (1)");
    run_filter(10000, 4, "4 threads");
    run_filter(10000, 8, "8 threads");
    run_filter(10000, -1, "Force serial (-1)");
    
    printf("\nNote: OMP_NUM_THREADS environment variable overrides cfg.num_threads=0\n");
#else
    printf("OpenMP is NOT enabled. Rebuild with USE_OPENMP=1 to test threading.\n");
    printf("Running serial version:\n\n");
    
    run_filter(10000, 0, "Serial");
#endif
    
    return 0;
}
