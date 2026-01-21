/**
 * @file test_zero_likelihood.c
 * @brief Test edge case where all particles have zero likelihood.
 */

#include "fastpf.h"
#include "../test_common.h"
#include <math.h>

/* Simple 1D state */
typedef struct {
    double x;
} state_t;

/* Model context */
typedef struct {
    double impossible_observation;
} model_ctx_t;

void prior_sample(void* ctx, void* x0_out, fastpf_rng_t* rng)
{
    state_t* s = (state_t*)x0_out;
    (void)ctx;
    s->x = fastpf_rng_normal(rng);
}

void transition_sample(void* ctx, const void* x_prev, void* x_out, fastpf_rng_t* rng)
{
    const state_t* s_prev = (const state_t*)x_prev;
    state_t* s = (state_t*)x_out;
    (void)ctx;
    s->x = s_prev->x + 0.1 * fastpf_rng_normal(rng);
}

double log_likelihood(void* ctx, const void* x, const void* y)
{
    const state_t* s = (const state_t*)x;
    model_ctx_t* m = (model_ctx_t*)ctx;
    double diff;
    
    (void)y;
    
    /* Create impossible scenario: observation is far from any particle */
    diff = s->x - m->impossible_observation;
    
    /* Return -INFINITY for all particles to trigger edge case */
    if (diff * diff > 1e-10) {
        return -INFINITY;
    }
    
    return -0.5 * diff * diff;
}

int test_zero_likelihood_edge_case(void)
{
    fastpf_t pf;
    fastpf_cfg_t cfg;
    fastpf_model_t model;
    model_ctx_t model_ctx;
    double observation;
    const fastpf_diagnostics_t* diag;
    const double* weights;
    size_t i;
    int status;
    double sum;
    
    printf("Testing zero-likelihood edge case...\n");
    
    /* Setup */
    model_ctx.impossible_observation = 1000.0; /* Far from any particle */
    observation = model_ctx.impossible_observation;
    
    fastpf_cfg_init(&cfg, 100, sizeof(state_t));
    cfg.rng_seed = 42;
    
    model.ctx = &model_ctx;
    model.prior_sample = prior_sample;
    model.transition_sample = transition_sample;
    model.log_likelihood = log_likelihood;
    model.rejuvenate = NULL;
    
    /* Initialize filter */
    status = fastpf_init(&pf, &cfg, &model);
    TEST_ASSERT(status == FASTPF_SUCCESS, "Failed to initialize PF");
    
    /* Step with impossible observation - all particles should get zero likelihood */
    status = fastpf_step(&pf, &observation);
    TEST_ASSERT(status == FASTPF_SUCCESS, "Failed to step PF");
    
    /* Check diagnostics */
    diag = fastpf_get_diagnostics(&pf);
    TEST_ASSERT(diag->log_norm_const == -INFINITY, 
                "log_norm_const should be -INFINITY when all likelihoods are zero");
    
    /* Check that weights are uniform (fallback behavior) */
    weights = fastpf_get_weights(&pf);
    sum = 0.0;
    for (i = 0; i < cfg.n_particles; i++) {
        TEST_ASSERT(!isnan(weights[i]), "Weight should not be NaN");
        TEST_ASSERT(weights[i] >= 0.0, "Weight should be non-negative");
        sum += weights[i];
    }
    
    TEST_ASSERT(fabs(sum - 1.0) < 1e-6, "Weights should sum to 1.0");
    
    /* Check that all weights are approximately equal (uniform) */
    for (i = 0; i < cfg.n_particles; i++) {
        TEST_ASSERT(fabs(weights[i] - 1.0/cfg.n_particles) < 1e-9,
                    "Weights should be uniform after zero-likelihood");
    }
    
    /* Cleanup */
    fastpf_free(&pf);
    
    printf("PASS: test_zero_likelihood_edge_case\n");
    return 0;
}

int main(void)
{
    printf("=== Zero Likelihood Edge Case Test ===\n");
    
    if (test_zero_likelihood_edge_case() != 0) {
        return 1;
    }
    
    printf("\n=== All zero-likelihood tests passed ===\n");
    return 0;
}
