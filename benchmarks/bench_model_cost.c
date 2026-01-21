/**
 * @file bench_model_cost.c
 * @brief Isolate user model cost vs library overhead
 */

#include "bench_common.h"

/* Null model functions for measuring library overhead */
void null_prior(void* ctx, void* x0_out, fastpf_rng_t* rng)
{
    state_t* s = (state_t*)x0_out;
    (void)ctx;
    (void)rng;
    s->x = 0.0;
}

void null_transition(void* ctx, const void* x_prev, void* x_out, fastpf_rng_t* rng)
{
    const state_t* s_prev = (const state_t*)x_prev;
    state_t* s = (state_t*)x_out;
    (void)ctx;
    (void)rng;
    s->x = s_prev->x;
}

double null_likelihood(void* ctx, const void* x, const void* y)
{
    (void)ctx;
    (void)x;
    (void)y;
    return 0.0;
}

static void run_test(const char* name, int N, int T, 
                     void (*prior)(void*, void*, fastpf_rng_t*),
                     void (*transition)(void*, const void*, void*, fastpf_rng_t*),
                     double (*likelihood)(void*, const void*, const void*),
                     model_params_t* params, double* observations)
{
    fastpf_cfg_t cfg;
    fastpf_model_t model;
    fastpf_t pf;
    double t_start, t_end;
    int k;
    
    fastpf_cfg_init(&cfg, N, sizeof(state_t));
    cfg.rng_seed = 42;
    cfg.resample_threshold = 0.5;
    
    model.ctx = params;
    model.prior_sample = prior;
    model.transition_sample = transition;
    model.log_likelihood = likelihood;
    model.rejuvenate = NULL;
    
    if (fastpf_init(&pf, &cfg, &model) != FASTPF_SUCCESS) {
        fprintf(stderr, "Init failed\n");
        return;
    }
    
    t_start = GET_TIME();
    
    for (k = 0; k < T; k++) {
        fastpf_step(&pf, &observations[k]);
    }
    
    t_end = GET_TIME();
    
    printf("%s,%d,%d,%.6f,%.6f\n", 
           name, N, T,
           (t_end - t_start) * 1e6,
           (t_end - t_start) * 1e6 / T);
    
    fastpf_free(&pf);
}

int main(int argc, char** argv)
{
    model_params_t params = {0.1, 0.5};
    double* x_true;
    double* observations;
    int N = 10000;
    int T = 1000;
    
    /* Parse command line args */
    if (argc > 1) {
        N = atoi(argv[1]);
    }
    if (argc > 2) {
        T = atoi(argv[2]);
    }
    
    /* Generate data */
    x_true = (double*)malloc(T * sizeof(double));
    observations = (double*)malloc(T * sizeof(double));
    generate_data(x_true, observations, T, params.process_std, params.obs_std, 12345);
    
    /* Header */
    printf("model,N,T,total_us,per_step_us\n");
    
    /* Library overhead only (null model) */
    run_test("null", N, T, null_prior, null_transition, null_likelihood, &params, observations);
    
    /* Full model */
    run_test("full", N, T, prior_sample, transition_sample, log_likelihood, &params, observations);
    
    free(x_true);
    free(observations);
    
    return 0;
}
