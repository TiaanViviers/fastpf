/**
 * @file bench_particle_scaling.c
 * @brief Measure performance scaling with particle count
 */

#include "bench_common.h"

static void run_test(int N, int T, model_params_t* params, double* observations)
{
    fastpf_cfg_t cfg;
    fastpf_model_t model;
    fastpf_t pf;
    const fastpf_diagnostics_t* diag;
    double t_start, t_end;
    int k, resample_count = 0;
    
    fastpf_cfg_init(&cfg, N, sizeof(state_t));
    cfg.rng_seed = 42;
    cfg.resample_threshold = 0.5;
    
    model.ctx = params;
    model.prior_sample = prior_sample;
    model.transition_sample = transition_sample;
    model.log_likelihood = log_likelihood;
    model.rejuvenate = NULL;
    
    if (fastpf_init(&pf, &cfg, &model) != FASTPF_SUCCESS) {
        fprintf(stderr, "Init failed\n");
        return;
    }
    
    t_start = GET_TIME();
    
    for (k = 0; k < T; k++) {
        fastpf_step(&pf, &observations[k]);
        diag = fastpf_get_diagnostics(&pf);
        if (diag->resampled) resample_count++;
    }
    
    t_end = GET_TIME();
    
    diag = fastpf_get_diagnostics(&pf);
    
    printf("%d,%d,%.6f,%.6f,%.3f,%d\n", 
           N, T,
           (t_end - t_start) * 1e6,
           (t_end - t_start) * 1e6 / T,
           diag->ess,
           resample_count);
    
    fastpf_free(&pf);
}

int main(int argc, char** argv)
{
    model_params_t params = {0.1, 0.5};
    double* x_true;
    double* observations;
    int T = 100;
    int N_values[] = {100, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000};
    int num_tests = sizeof(N_values) / sizeof(N_values[0]);
    int i;
    
    /* Parse command line args */
    if (argc > 1) {
        T = atoi(argv[1]);
    }
    
    /* Generate data */
    x_true = (double*)malloc(T * sizeof(double));
    observations = (double*)malloc(T * sizeof(double));
    generate_data(x_true, observations, T, params.process_std, params.obs_std, 12345);
    
    /* Header */
    printf("N,T,total_us,per_step_us,ess,resamples\n");
    
    /* Run tests */
    for (i = 0; i < num_tests; i++) {
        run_test(N_values[i], T, &params, observations);
    }
    
    free(x_true);
    free(observations);
    
    return 0;
}
