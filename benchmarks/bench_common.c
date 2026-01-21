/**
 * @file bench_common.c
 * @brief Common utilities implementation
 */

#include "bench_common.h"

void prior_sample(void* ctx, void* x0_out, fastpf_rng_t* rng)
{
    state_t* s = (state_t*)x0_out;
    (void)ctx;
    s->x = fastpf_rng_normal(rng);
}

void transition_sample(void* ctx, const void* x_prev, void* x_out, fastpf_rng_t* rng)
{
    const model_params_t* params = (const model_params_t*)ctx;
    const state_t* s_prev = (const state_t*)x_prev;
    state_t* s = (state_t*)x_out;
    
    s->x = s_prev->x + params->process_std * fastpf_rng_normal(rng);
}

double log_likelihood(void* ctx, const void* x, const void* y)
{
    const model_params_t* params = (const model_params_t*)ctx;
    const state_t* s = (const state_t*)x;
    const double* obs = (const double*)y;
    double diff = s->x - *obs;
    double var = params->obs_std * params->obs_std;
    
    return -0.5 * (diff * diff / var + log(2.0 * 3.14159265358979323846 * var));
}

void generate_data(double* x_true, double* observations, int T, 
                   double process_std, double obs_std, uint64_t seed)
{
    fastpf_rng_t rng;
    int k;
    
    fastpf_rng_seed(&rng, seed);
    
    x_true[0] = fastpf_rng_normal(&rng);
    observations[0] = x_true[0] + obs_std * fastpf_rng_normal(&rng);
    
    for (k = 1; k < T; k++) {
        x_true[k] = x_true[k-1] + process_std * fastpf_rng_normal(&rng);
        observations[k] = x_true[k] + obs_std * fastpf_rng_normal(&rng);
    }
}
