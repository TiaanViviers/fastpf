/**
 * @file example_random_walk.c
 * @brief Simple example: 1D random walk with noisy observations.
 *
 * Demonstrates how to use the fastpf library to track a simple state-space model.
 */

#include "fastpf.h"

/* For this example, we include the internal header to enable stack allocation.
 * Production code can either:
 *   1. Use heap allocation (malloc/free) with opaque pointers, OR
 *   2. Include fastpf_internal.h if stack allocation is required.
 * The structs are kept opaque in the public API to allow implementation changes. */
#include "../src/fastpf_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* ========================================================================
 * Model: 1D random walk
 *   x_k = x_{k-1} + w_k,  w_k ~ N(0, 0.3^2)
 *   y_k = x_k + v_k,      v_k ~ N(0, 0.5^2)
 * ======================================================================== */

typedef struct {
    double process_noise_std;
    double obs_noise_std;
} model_params_t;

typedef struct {
    double x;
} state_t;

typedef struct {
    double y;
} observation_t;

/* Prior: x_0 ~ N(0, 1) */
void prior_sample(void* ctx, void* x0_out, fastpf_rng_t* rng)
{
    state_t* state = (state_t*)x0_out;
    (void)ctx; /* Unused */
    
    state->x = fastpf_rng_normal(rng);
}

/* Transition: x_k = x_{k-1} + N(0, sigma_x^2) */
void transition_sample(void* ctx, const void* x_prev, void* x_out, fastpf_rng_t* rng)
{
    model_params_t* params = (model_params_t*)ctx;
    const state_t* prev = (const state_t*)x_prev;
    state_t* next = (state_t*)x_out;
    
    next->x = prev->x + params->process_noise_std * fastpf_rng_normal(rng);
}

/* Likelihood: p(y_k | x_k) = N(x_k, sigma_y^2) */
double log_likelihood(void* ctx, const void* x, const void* y)
{
    model_params_t* params = (model_params_t*)ctx;
    const state_t* state = (const state_t*)x;
    const observation_t* obs = (const observation_t*)y;
    double residual, log_lik;
    
    residual = obs->y - state->x;
    log_lik = -0.5 * log(2.0 * 3.14159265358979323846 * params->obs_noise_std * params->obs_noise_std)
              -0.5 * (residual * residual) / (params->obs_noise_std * params->obs_noise_std);
    
    return log_lik;
}

/* ========================================================================
 * Helper: compute weighted mean and std dev
 * ======================================================================== */
void compute_pf_stats(const fastpf_t* pf, double* mean_out, double* std_out)
{
    size_t i, n;
    double mean, variance, diff;
    const double* weights;
    const state_t* state;
    
    n = fastpf_num_particles(pf);
    weights = fastpf_get_weights(pf);
    
    /* Compute weighted mean */
    mean = 0.0;
    for (i = 0; i < n; i++) {
        state = (const state_t*)fastpf_get_particle(pf, i);
        mean += weights[i] * state->x;
    }
    
    /* Compute weighted variance */
    variance = 0.0;
    for (i = 0; i < n; i++) {
        state = (const state_t*)fastpf_get_particle(pf, i);
        diff = state->x - mean;
        variance += weights[i] * diff * diff;
    }
    
    *mean_out = mean;
    *std_out = sqrt(variance);
}

/* ========================================================================
 * Main example
 * ======================================================================== */
int main(void)
{
    model_params_t params;
    fastpf_cfg_t cfg;
    fastpf_model_t model;
    fastpf_t pf;
    fastpf_rng_t data_rng;
    
    observation_t observations[30];
    double x_true[30];
    int k, T;
    int status;
    
    double pf_mean, pf_std;
    const fastpf_diagnostics_t* diag;
    
    printf("=== fastpf Example: 1D Random Walk ===\n\n");
    
    /* Setup model parameters */
    params.process_noise_std = 0.3;
    params.obs_noise_std = 0.5;
    
    /* Generate synthetic data */
    T = 30;
    fastpf_rng_seed(&data_rng, 12345ULL);
    
    x_true[0] = fastpf_rng_normal(&data_rng);
    observations[0].y = x_true[0] + params.obs_noise_std * fastpf_rng_normal(&data_rng);
    
    for (k = 1; k < T; k++) {
        x_true[k] = x_true[k-1] + params.process_noise_std * fastpf_rng_normal(&data_rng);
        observations[k].y = x_true[k] + params.obs_noise_std * fastpf_rng_normal(&data_rng);
    }
    
    printf("Generated %d observations from 1D random walk model.\n", T);
    printf("  Process noise std: %.2f\n", params.process_noise_std);
    printf("  Observation noise std: %.2f\n\n", params.obs_noise_std);
    
    /* Configure particle filter */
    fastpf_cfg_init(&cfg, 500, sizeof(state_t));
    cfg.rng_seed = 42ULL;
    cfg.resample_threshold = 0.5;
    cfg.resample_method = FASTPF_RESAMPLE_SYSTEMATIC;
    
    model.ctx = &params;
    model.prior_sample = prior_sample;
    model.transition_sample = transition_sample;
    model.log_likelihood = log_likelihood;
    model.rejuvenate = NULL;
    
    /* Initialize particle filter */
    status = fastpf_init(&pf, &cfg, &model);
    if (status != FASTPF_SUCCESS) {
        fprintf(stderr, "Error: Failed to initialize particle filter (code %d)\n", status);
        return 1;
    }
    
    printf("Initialized particle filter with %lu particles.\n\n",
           (unsigned long)cfg.n_particles);
    printf("Running particle filter...\n");
    printf("%-4s  %-8s  %-8s  %-10s  %-10s  %-6s  %-8s  %s\n",
           "k", "y_k", "x_true", "PF_mean", "PF_std", "ESS", "MaxWt", "Resamp?");
    printf("---------------------------------------------------------------------\n");
    
    /* Run particle filter */
    for (k = 0; k < T; k++) {
        /* Filter step */
        status = fastpf_step(&pf, &observations[k]);
        if (status != FASTPF_SUCCESS) {
            fprintf(stderr, "Error: PF step %d failed (code %d)\n", k, status);
            fastpf_free(&pf);
            return 1;
        }
        
        /* Get results */
        compute_pf_stats(&pf, &pf_mean, &pf_std);
        diag = fastpf_get_diagnostics(&pf);
        
        /* Print results */
        printf("%-4d  %8.3f  %8.3f  %10.3f  %10.3f  %6.1f  %8.5f  %s\n",
               k,
               observations[k].y,
               x_true[k],
               pf_mean,
               pf_std,
               diag->ess,
               diag->max_weight,
               diag->resampled ? "YES" : "NO");
    }
    
    printf("\n=== Filtering complete ===\n");
    
    /* Cleanup */
    fastpf_free(&pf);
    
    return 0;
}
