/**
 * @file test_lgssm.c
 * @brief Accuracy test: PF vs Kalman filter oracle on Linear-Gaussian SSM.
 *
 * Tests the particle filter against exact Kalman filter on a 1D random walk model:
 *   x_{k} = x_{k-1} + w_k,  w_k ~ N(0, sigma_x^2)
 *   y_{k} = x_k + v_k,      v_k ~ N(0, sigma_y^2)
 */

#include "fastpf.h"
#include "../test_common.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

/* ========================================================================
 * Model parameters
 * ======================================================================== */

typedef struct {
    double sigma_x; /* Process noise std dev */
    double sigma_y; /* Observation noise std dev */
    double x0_mean; /* Prior mean */
    double x0_std;  /* Prior std dev */
} lgssm_params_t;

/* ========================================================================
 * Model state (single double for 1D case)
 * ======================================================================== */

typedef struct {
    double x;
} state_1d_t;

typedef struct {
    double y;
} obs_1d_t;

/* ========================================================================
 * PF model callbacks for LGSSM
 * ======================================================================== */

void lgssm_prior_sample(void* ctx, void* x0_out, fastpf_rng_t* rng)
{
    lgssm_params_t* params = (lgssm_params_t*)ctx;
    state_1d_t* state = (state_1d_t*)x0_out;
    
    state->x = params->x0_mean + params->x0_std * fastpf_rng_normal(rng);
}

void lgssm_transition_sample(void* ctx, const void* x_prev, void* x_out, fastpf_rng_t* rng)
{
    lgssm_params_t* params = (lgssm_params_t*)ctx;
    const state_1d_t* prev = (const state_1d_t*)x_prev;
    state_1d_t* next = (state_1d_t*)x_out;
    
    next->x = prev->x + params->sigma_x * fastpf_rng_normal(rng);
}

double lgssm_log_likelihood(void* ctx, const void* x, const void* y)
{
    lgssm_params_t* params = (lgssm_params_t*)ctx;
    const state_1d_t* state = (const state_1d_t*)x;
    const obs_1d_t* obs = (const obs_1d_t*)y;
    double residual, log_lik;
    
    residual = obs->y - state->x;
    log_lik = -0.5 * log(2.0 * 3.14159265358979323846 * params->sigma_y * params->sigma_y)
              -0.5 * (residual * residual) / (params->sigma_y * params->sigma_y);
    
    return log_lik;
}

/* ========================================================================
 * Kalman filter (oracle) for 1D LGSSM
 * ======================================================================== */

typedef struct {
    double mean;
    double var;
} kf_state_t;

void kf_init(kf_state_t* kf, double x0_mean, double x0_var)
{
    kf->mean = x0_mean;
    kf->var = x0_var;
}

void kf_predict(kf_state_t* kf, double sigma_x)
{
    /* Prediction: x_{k|k-1} ~ N(x_{k-1|k-1}, P_{k-1|k-1} + sigma_x^2) */
    kf->var += sigma_x * sigma_x;
}

void kf_update(kf_state_t* kf, double y, double sigma_y)
{
    double K, innovation, S;
    
    /* Innovation covariance: S = P_{k|k-1} + sigma_y^2 */
    S = kf->var + sigma_y * sigma_y;
    
    /* Kalman gain: K = P_{k|k-1} / S */
    K = kf->var / S;
    
    /* Innovation: y - x_{k|k-1} */
    innovation = y - kf->mean;
    
    /* Update: x_{k|k} = x_{k|k-1} + K * innovation */
    kf->mean += K * innovation;
    
    /* Update covariance: P_{k|k} = (1 - K) * P_{k|k-1} */
    kf->var = (1.0 - K) * kf->var;
}

/* ========================================================================
 * Helper: compute weighted mean from PF
 * ======================================================================== */
double pf_weighted_mean(const fastpf_t* pf)
{
    size_t i, n;
    double mean;
    const double* weights;
    const state_1d_t* state;
    
    n = fastpf_num_particles(pf);
    weights = fastpf_get_weights(pf);
    
    mean = 0.0;
    for (i = 0; i < n; i++) {
        state = (const state_1d_t*)fastpf_get_particle(pf, i);
        mean += weights[i] * state->x;
    }
    
    return mean;
}

/* ========================================================================
 * Test: PF vs KF on a short synthetic trajectory
 * ======================================================================== */
int test_pf_vs_kf_1d(void)
{
    lgssm_params_t params;
    fastpf_cfg_t cfg;
    fastpf_model_t model;
    fastpf_t pf;
    kf_state_t kf;
    fastpf_rng_t data_rng;
    
    double x_true[50];
    obs_1d_t obs[50];
    int T, k;
    double rmse, diff;
    double pf_mean;
    int status;
    
    /* Model parameters */
    params.sigma_x = 0.5;
    params.sigma_y = 1.0;
    params.x0_mean = 0.0;
    params.x0_std = 1.0;
    
    /* Generate synthetic data */
    T = 50;
    fastpf_rng_seed(&data_rng, 12345ULL);
    
    x_true[0] = params.x0_mean + params.x0_std * fastpf_rng_normal(&data_rng);
    obs[0].y = x_true[0] + params.sigma_y * fastpf_rng_normal(&data_rng);
    
    for (k = 1; k < T; k++) {
        x_true[k] = x_true[k-1] + params.sigma_x * fastpf_rng_normal(&data_rng);
        obs[k].y = x_true[k] + params.sigma_y * fastpf_rng_normal(&data_rng);
    }
    
    /* Initialize particle filter */
    fastpf_cfg_init(&cfg, 1000, sizeof(state_1d_t));
    cfg.rng_seed = 99999ULL;
    cfg.resample_threshold = 0.5;
    
    model.ctx = &params;
    model.prior_sample = lgssm_prior_sample;
    model.transition_sample = lgssm_transition_sample;
    model.log_likelihood = lgssm_log_likelihood;
    model.rejuvenate = NULL;
    
    status = fastpf_init(&pf, &cfg, &model);
    if (status != FASTPF_SUCCESS) {
        fprintf(stderr, "FAIL: PF initialization failed\n");
        return 1;
    }
    
    /* Initialize Kalman filter */
    kf_init(&kf, params.x0_mean, params.x0_std * params.x0_std);
    
    /* Run both filters */
    rmse = 0.0;
    for (k = 0; k < T; k++) {
        /* PF step */
        status = fastpf_step(&pf, &obs[k]);
        if (status != FASTPF_SUCCESS) {
            fprintf(stderr, "FAIL: PF step %d failed\n", k);
            fastpf_free(&pf);
            return 1;
        }
        
        /* KF step */
        kf_predict(&kf, params.sigma_x);
        kf_update(&kf, obs[k].y, params.sigma_y);
        
        /* Compare estimates */
        pf_mean = pf_weighted_mean(&pf);
        diff = pf_mean - kf.mean;
        rmse += diff * diff;
        
        /* Debug: print first few steps */
        if (k < 5) {
            printf("  k=%2d: x_true=%.3f, y=%.3f, KF=%.3f, PF=%.3f, diff=%.4f\n",
                   k, x_true[k], obs[k].y, kf.mean, pf_mean, diff);
        }
    }
    
    rmse = sqrt(rmse / T);
    
    printf("  RMSE(PF - KF) over %d steps: %.6f\n", T, rmse);
    
    /* RMSE should be small (PF with 1000 particles should be close to KF) */
    if (rmse > 0.5) {
        fprintf(stderr, "FAIL: RMSE too large (%.6f > 0.5)\n", rmse);
        fastpf_free(&pf);
        return 1;
    }
    
    printf("PASS: test_pf_vs_kf_1d (RMSE=%.6f)\n", rmse);
    
    fastpf_free(&pf);
    return 0;
}

/* ========================================================================
 * Test: PF diagnostics checks
 * ======================================================================== */
int test_pf_diagnostics(void)
{
    lgssm_params_t params;
    fastpf_cfg_t cfg;
    fastpf_model_t model;
    fastpf_t pf;
    obs_1d_t obs;
    const fastpf_diagnostics_t* diag;
    int status;
    
    /* Model parameters */
    params.sigma_x = 0.5;
    params.sigma_y = 1.0;
    params.x0_mean = 0.0;
    params.x0_std = 1.0;
    
    /* Initialize PF */
    fastpf_cfg_init(&cfg, 500, sizeof(state_1d_t));
    cfg.rng_seed = 42ULL;
    
    model.ctx = &params;
    model.prior_sample = lgssm_prior_sample;
    model.transition_sample = lgssm_transition_sample;
    model.log_likelihood = lgssm_log_likelihood;
    model.rejuvenate = NULL;
    
    status = fastpf_init(&pf, &cfg, &model);
    if (status != FASTPF_SUCCESS) {
        fprintf(stderr, "FAIL: PF initialization failed\n");
        return 1;
    }
    
    /* Run a few steps */
    obs.y = 1.0;
    fastpf_step(&pf, &obs);
    obs.y = 1.5;
    fastpf_step(&pf, &obs);
    obs.y = 2.0;
    fastpf_step(&pf, &obs);
    
    /* Check diagnostics */
    diag = fastpf_get_diagnostics(&pf);
    
    if (diag->ess < 1.0 || diag->ess > 500.0) {
        fprintf(stderr, "FAIL: ESS out of range: %.2f\n", diag->ess);
        fastpf_free(&pf);
        return 1;
    }
    
    if (diag->max_weight < 0.0 || diag->max_weight > 1.0) {
        fprintf(stderr, "FAIL: max_weight out of range: %.6f\n", diag->max_weight);
        fastpf_free(&pf);
        return 1;
    }
    
    printf("  ESS=%.2f, max_weight=%.6f, resampled=%d\n",
           diag->ess, diag->max_weight, diag->resampled);
    
    printf("PASS: test_pf_diagnostics\n");
    
    fastpf_free(&pf);
    return 0;
}

/* ========================================================================
 * Main test runner
 * ======================================================================== */
int main(void)
{
    int result = 0;
    
    printf("=== LGSSM Accuracy Tests (PF vs Kalman Filter) ===\n\n");
    
    printf("Test 1: PF vs KF on 1D random walk\n");
    result |= test_pf_vs_kf_1d();
    
    printf("\nTest 2: PF diagnostics\n");
    result |= test_pf_diagnostics();
    
    if (result == 0) {
        printf("\n=== All accuracy tests passed ===\n");
    } else {
        printf("\n=== Some accuracy tests failed ===\n");
    }
    
    return result;
}
