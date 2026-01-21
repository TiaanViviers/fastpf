/**
 * @file fastpf.c
 * @brief Core particle filter implementation: bootstrap SIR.
 */

#include "fastpf.h"
#include "fastpf_internal.h"
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <float.h>

#ifdef FASTPF_USE_OPENMP
    #include <omp.h>
#endif

/* C90-compatible infinity */
#ifndef INFINITY
    #define INFINITY (1.0 / 0.0)
#endif

/* C90-compatible isnan/isinf (may be macros in C99+) */
#ifndef isnan
    #define isnan(x) ((x) != (x))
#endif
#ifndef isinf
    #define isinf(x) (!isnan(x) && isnan((x) - (x)))
#endif

/* Forward declarations of internal functions */
extern double fastpf_logsumexp(const double* log_weights, size_t n);
extern double fastpf_normalize_weights(double* log_weights, double* norm_weights, size_t n);
extern double fastpf_compute_ess(const double* norm_weights, size_t n);
extern double fastpf_max(const double* arr, size_t n);
extern void fastpf_resample(fastpf_resample_method_t method,
                             const double* norm_weights,
                             size_t n,
                             size_t* indices,
                             fastpf_rng_t* rng);

/* Macro for accessing particle i in a contiguous array */
#define PARTICLE_PTR(base, i, state_size) ((void*)((base) + (i) * (state_size)))

/* ========================================================================
 * Configuration helpers
 * ======================================================================== */

void fastpf_cfg_init(fastpf_cfg_t* cfg, size_t n_particles, size_t state_size)
{
    assert(cfg != NULL);
    assert(n_particles > 0);
    assert(state_size > 0);
    
    cfg->n_particles = n_particles;
    cfg->state_size = state_size;
    cfg->rng_seed = 42; /* Default seed */
    cfg->resample_threshold = 0.5; /* Resample when ESS/N < 0.5 */
    cfg->resample_method = FASTPF_RESAMPLE_SYSTEMATIC;
    cfg->num_threads = 0; /* 0 = auto-detect (use omp_get_max_threads) */
    cfg->flags = 0; /* Reserved for future use */
}

/* ========================================================================
 * Initialization and cleanup
 * ======================================================================== */

int fastpf_init(fastpf_t* pf, const fastpf_cfg_t* cfg, const fastpf_model_t* model)
{
    size_t i;
    size_t total_particle_bytes;
    void* particle_ptr;
    
    assert(pf != NULL);
    assert(cfg != NULL);
    assert(model != NULL);
    assert(cfg->n_particles > 0);
    assert(cfg->state_size > 0);
    assert(model->prior_sample != NULL);
    assert(model->transition_sample != NULL);
    assert(model->log_likelihood != NULL);
    
    /* Zero out the structure */
    memset(pf, 0, sizeof(fastpf_t));
    
    /* Copy configuration and model */
    pf->cfg = *cfg;
    pf->model = *model;
    
    /* Initialize RNG */
    fastpf_rng_seed(&pf->rng, cfg->rng_seed);
    
    /* Initialize per-thread RNGs for OpenMP (if enabled) */
    pf->thread_rngs = NULL;
#ifdef FASTPF_USE_OPENMP
    {
        int num_threads;
        int tid;
        
        /* Set thread count based on configuration:
         *   num_threads = 0  : Auto (use omp_get_max_threads)
         *   num_threads > 0  : Explicit count
         *   num_threads = -1 : Force serial (set to 1)
         */
        if (cfg->num_threads == 0) {
            num_threads = omp_get_max_threads();
        } else if (cfg->num_threads < 0) {
            num_threads = 1;
            omp_set_num_threads(1);
        } else {
            num_threads = cfg->num_threads;
            omp_set_num_threads(cfg->num_threads);
        }
        
        pf->thread_rngs = (fastpf_rng_t*)malloc(num_threads * sizeof(fastpf_rng_t));
        if (pf->thread_rngs == NULL) {
            return FASTPF_ERR_ALLOC;
        }
        
        /* Seed each thread RNG with a different stream derived from master seed */
        for (tid = 0; tid < num_threads; tid++) {
            fastpf_rng_seed(&pf->thread_rngs[tid], cfg->rng_seed + (uint64_t)tid + 1ULL);
        }
    }
#endif
    
    /* Allocate particle storage (double buffered) */
    total_particle_bytes = cfg->n_particles * cfg->state_size;
    pf->particles_curr = (unsigned char*)malloc(total_particle_bytes);
    pf->particles_next = (unsigned char*)malloc(total_particle_bytes);
    
    if (pf->particles_curr == NULL || pf->particles_next == NULL) {
        fastpf_free(pf);
        return FASTPF_ERR_ALLOC;
    }
    
    /* Allocate weight storage */
    pf->log_weights = (double*)malloc(cfg->n_particles * sizeof(double));
    pf->norm_weights = (double*)malloc(cfg->n_particles * sizeof(double));
    
    if (pf->log_weights == NULL || pf->norm_weights == NULL) {
        fastpf_free(pf);
        return FASTPF_ERR_ALLOC;
    }
    
    /* Allocate resampling scratch */
    pf->resample_indices = (size_t*)malloc(cfg->n_particles * sizeof(size_t));
    if (pf->resample_indices == NULL) {
        fastpf_free(pf);
        return FASTPF_ERR_ALLOC;
    }
    
    /* Initialize particles from prior */
#ifdef FASTPF_USE_OPENMP
    #pragma omp parallel for private(particle_ptr)
#endif
    for (i = 0; i < cfg->n_particles; i++) {
        fastpf_rng_t* rng_ptr;
        particle_ptr = PARTICLE_PTR(pf->particles_curr, i, cfg->state_size);
        
#ifdef FASTPF_USE_OPENMP
        rng_ptr = &pf->thread_rngs[omp_get_thread_num()];
#else
        rng_ptr = &pf->rng;
#endif
        model->prior_sample(model->ctx, particle_ptr, rng_ptr);
    }
    
    /* Initialize weights to uniform: log(1/N) */
    for (i = 0; i < cfg->n_particles; i++) {
        pf->log_weights[i] = -log((double)cfg->n_particles);
        pf->norm_weights[i] = 1.0 / (double)cfg->n_particles;
    }
    
    /* Initialize diagnostics */
    pf->diag.ess = (double)cfg->n_particles;
    pf->diag.max_weight = 1.0 / (double)cfg->n_particles;
    pf->diag.log_norm_const = 0.0;
    pf->diag.resampled = 0;
    
    pf->initialized = 1;
    
    return FASTPF_SUCCESS;
}

void fastpf_free(fastpf_t* pf)
{
    if (pf == NULL) {
        return;
    }
    
    free(pf->particles_curr);
    free(pf->particles_next);
    free(pf->log_weights);
    free(pf->norm_weights);
    free(pf->resample_indices);
    free(pf->thread_rngs);  /* Safe even if NULL */
    
    memset(pf, 0, sizeof(fastpf_t));
}

/* ========================================================================
 * Core filtering step
 * ======================================================================== */

int fastpf_step(fastpf_t* pf, const void* y_k)
{
    size_t i;
    size_t n;
    size_t state_size;
    const void* x_prev;
    void* x_next;
    double log_lik;
    double log_weight_sum;
    double ess, ess_threshold;
    int do_resample;
    unsigned char* temp_ptr;
    
    assert(pf != NULL);
    assert(pf->initialized);
    assert(y_k != NULL);
    
    n = pf->cfg.n_particles;
    state_size = pf->cfg.state_size;
    
    /* -----------------------------------------------------------------------
     * Step 1: Propagate particles through transition model
     * ----------------------------------------------------------------------- */
#ifdef FASTPF_USE_OPENMP
    #pragma omp parallel for private(x_prev, x_next)
#endif
    for (i = 0; i < n; i++) {
        fastpf_rng_t* rng_ptr;
        x_prev = PARTICLE_PTR(pf->particles_curr, i, state_size);
        x_next = PARTICLE_PTR(pf->particles_next, i, state_size);
        
#ifdef FASTPF_USE_OPENMP
        rng_ptr = &pf->thread_rngs[omp_get_thread_num()];
#else
        rng_ptr = &pf->rng;
#endif
        pf->model.transition_sample(pf->model.ctx, x_prev, x_next, rng_ptr);
    }
    
    /* Swap buffers: next becomes current */
    temp_ptr = pf->particles_curr;
    pf->particles_curr = pf->particles_next;
    pf->particles_next = temp_ptr;
    
    /* -----------------------------------------------------------------------
     * Step 2: Update weights with likelihood
     * ----------------------------------------------------------------------- */
#ifdef FASTPF_USE_OPENMP
    #pragma omp parallel for private(x_prev, log_lik)
#endif
    for (i = 0; i < n; i++) {
        x_prev = PARTICLE_PTR(pf->particles_curr, i, state_size);
        log_lik = pf->model.log_likelihood(pf->model.ctx, x_prev, y_k);
        
        assert(!isnan(log_lik));
        assert(!isinf(log_lik) || log_lik == -INFINITY);
        
        /* Update log-weight: log(w_k) = log(w_{k-1}) + log(p(y_k | x_k)) */
        pf->log_weights[i] += log_lik;
    }
    
    /* -----------------------------------------------------------------------
     * Step 3: Normalize weights
     * ----------------------------------------------------------------------- */
    log_weight_sum = fastpf_normalize_weights(pf->log_weights, pf->norm_weights, n);
    
    /* -----------------------------------------------------------------------
     * Step 4: Compute diagnostics
     * ----------------------------------------------------------------------- */
    ess = fastpf_compute_ess(pf->norm_weights, n);
    pf->diag.ess = ess;
    pf->diag.max_weight = fastpf_max(pf->norm_weights, n);
    pf->diag.log_norm_const = log_weight_sum;
    
    /* -----------------------------------------------------------------------
     * Step 5: Adaptive resampling
     * ----------------------------------------------------------------------- */
    ess_threshold = pf->cfg.resample_threshold * (double)n;
    do_resample = (ess < ess_threshold);
    pf->diag.resampled = do_resample;
    
    if (do_resample) {
        fastpf_resample(pf->cfg.resample_method,
                        pf->norm_weights,
                        n,
                        pf->resample_indices,
                        &pf->rng);
        
        /* Copy resampled particles from curr to next buffer */
        for (i = 0; i < n; i++) {
            size_t src_idx = pf->resample_indices[i];
            const void* src = PARTICLE_PTR(pf->particles_curr, src_idx, state_size);
            void* dst = PARTICLE_PTR(pf->particles_next, i, state_size);
            memcpy(dst, src, state_size);
        }
        
        /* Swap buffers again: resampled particles become current */
        temp_ptr = pf->particles_curr;
        pf->particles_curr = pf->particles_next;
        pf->particles_next = temp_ptr;
        
        /* Reset weights to uniform */
        for (i = 0; i < n; i++) {
            pf->log_weights[i] = -log((double)n);
            pf->norm_weights[i] = 1.0 / (double)n;
        }
        
        /* Optional: rejuvenation/roughening */
        if (pf->model.rejuvenate != NULL) {
            for (i = 0; i < n; i++) {
                void* particle = PARTICLE_PTR(pf->particles_curr, i, state_size);
                pf->model.rejuvenate(pf->model.ctx, particle, &pf->rng);
            }
        }
        
        /* Update ESS after resampling */
        pf->diag.ess = (double)n;
        pf->diag.max_weight = 1.0 / (double)n;
    }
    
    return FASTPF_SUCCESS;
}

/* ========================================================================
 * Accessors
 * ======================================================================== */

const void* fastpf_get_particle(const fastpf_t* pf, size_t i)
{
    assert(pf != NULL);
    assert(pf->initialized);
    
    if (i >= pf->cfg.n_particles) {
        return NULL;
    }
    
    return PARTICLE_PTR(pf->particles_curr, i, pf->cfg.state_size);
}

const double* fastpf_get_weights(const fastpf_t* pf)
{
    assert(pf != NULL);
    assert(pf->initialized);
    
    return pf->norm_weights;
}

const fastpf_diagnostics_t* fastpf_get_diagnostics(const fastpf_t* pf)
{
    assert(pf != NULL);
    assert(pf->initialized);
    
    return &pf->diag;
}

size_t fastpf_num_particles(const fastpf_t* pf)
{
    assert(pf != NULL);
    assert(pf->initialized);
    
    return pf->cfg.n_particles;
}

size_t fastpf_state_size(const fastpf_t* pf)
{
    assert(pf != NULL);
    assert(pf->initialized);
    
    return pf->cfg.state_size;
}
