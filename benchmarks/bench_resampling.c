/**
 * @file bench_resampling.c
 * @brief Measure resampling overhead by varying threshold
 */

#include "bench_common.h"

static void run_test(int N, int T, double threshold, model_params_t* params, double* observations)
{
    fastpf_cfg_t cfg;
    fastpf_model_t model;
    fastpf_t pf;
    const fastpf_diagnostics_t* diag;
    double t_start, t_end;
    int k, resample_count = 0;
    
    fastpf_cfg_init(&cfg, N, sizeof(state_t));
    cfg.rng_seed = 42;
    cfg.resample_threshold = threshold;
    
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
    
    printf("%d,%d,%.2f,%.6f,%.6f,%.3f,%d\n", 
           N, T, threshold,
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
    int N = 10000;
    int T = 1000;
    double thresholds[] = {0.0, 0.3, 0.5, 0.7, 0.9, 1.0};
    int num_tests = sizeof(thresholds) / sizeof(thresholds[0]);
    int i;
    
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
    printf("N,T,threshold,total_us,per_step_us,ess,resamples\n");
    
    /* Run tests */
    for (i = 0; i < num_tests; i++) {
        run_test(N, T, thresholds[i], &params, observations);
    }
    
    free(x_true);
    free(observations);
    
    return 0;
}
