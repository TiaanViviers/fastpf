/**
 * @file resample.c
 * @brief Resampling algorithms for particle filters.
 *
 * Implements systematic and stratified resampling with O(N) complexity.
 */

#include "fastpf.h"
#include "fastpf_internal.h"
#include <assert.h>

/* ========================================================================
 * Systematic resampling
 * ======================================================================== */

/**
 * @brief Systematic resampling: low-variance, deterministic given u.
 *
 * Algorithm:
 *   1. Draw u ~ Uniform(0, 1/N)
 *   2. For i = 0, ..., N-1: resample particle k where (u + i/N) falls in CDF
 *
 * @param norm_weights Normalized weights (must sum to 1).
 * @param n Number of particles.
 * @param indices Output array of resampled indices.
 * @param rng RNG state.
 */
void fastpf_resample_systematic(const double* norm_weights, size_t n, size_t* indices, fastpf_rng_t* rng)
{
    size_t i, j;
    double u, cumsum, step, pos;
    
    assert(norm_weights != NULL);
    assert(indices != NULL);
    assert(n > 0);
    
    /* Draw initial offset u ~ Uniform(0, 1/N) */
    u = fastpf_rng_uniform01(rng) / (double)n;
    step = 1.0 / (double)n;
    
    cumsum = norm_weights[0];
    j = 0;
    
    for (i = 0; i < n; i++) {
        pos = u + i * step;
        
        /* Advance j until cumsum >= pos */
        while (pos > cumsum && j < n - 1) {
            j++;
            cumsum += norm_weights[j];
        }
        
        indices[i] = j;
    }
    
    /* Debug: all indices should be valid */
    for (i = 0; i < n; i++) {
        assert(indices[i] < n);
    }
}

/**
 * @brief Stratified resampling: similar to systematic but with per-stratum jitter.
 *
 * Algorithm:
 *   For i = 0, ..., N-1: draw u_i ~ Uniform(i/N, (i+1)/N), resample where u_i falls in CDF.
 *
 * @param norm_weights Normalized weights (must sum to 1).
 * @param n Number of particles.
 * @param indices Output array of resampled indices.
 * @param rng RNG state.
 */
void fastpf_resample_stratified(const double* norm_weights, size_t n, size_t* indices, fastpf_rng_t* rng)
{
    size_t i, j;
    double cumsum, step, u_i, pos;
    
    assert(norm_weights != NULL);
    assert(indices != NULL);
    assert(n > 0);
    
    step = 1.0 / (double)n;
    cumsum = norm_weights[0];
    j = 0;
    
    for (i = 0; i < n; i++) {
        /* Draw u_i ~ Uniform(i/N, (i+1)/N) */
        u_i = (i + fastpf_rng_uniform01(rng)) * step;
        pos = u_i;
        
        /* Advance j until cumsum >= pos */
        while (pos > cumsum && j < n - 1) {
            j++;
            cumsum += norm_weights[j];
        }
        
        indices[i] = j;
    }
    
    /* Debug: all indices should be valid */
    for (i = 0; i < n; i++) {
        assert(indices[i] < n);
    }
}

/**
 * @brief Dispatch to the appropriate resampling method.
 * @param method Resampling method enum.
 * @param norm_weights Normalized weights.
 * @param n Number of particles.
 * @param indices Output array.
 * @param rng RNG state.
 */
void fastpf_resample(fastpf_resample_method_t method,
                     const double* norm_weights,
                     size_t n,
                     size_t* indices,
                     fastpf_rng_t* rng)
{
    switch (method) {
        case FASTPF_RESAMPLE_SYSTEMATIC:
            fastpf_resample_systematic(norm_weights, n, indices, rng);
            break;
        case FASTPF_RESAMPLE_STRATIFIED:
            fastpf_resample_stratified(norm_weights, n, indices, rng);
            break;
        default:
            assert(0 && "Unknown resampling method");
            break;
    }
}
