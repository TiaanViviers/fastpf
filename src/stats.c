/**
 * @file stats.c
 * @brief Statistical utilities: ESS, weighted moments, diagnostics.
 */

#include "fastpf.h"
#include "fastpf_internal.h"
#include <assert.h>

/* ========================================================================
 * Effective Sample Size (ESS)
 * ======================================================================== */

/**
 * @brief Compute effective sample size: ESS = 1 / sum(w_i^2).
 * @param norm_weights Normalized weights (must sum to 1).
 * @param n Number of particles.
 * @return ESS value in [1, n].
 */
double fastpf_compute_ess(const double* norm_weights, size_t n)
{
    size_t i;
    double sum_sq;
    
    assert(norm_weights != NULL);
    assert(n > 0);
    
    sum_sq = 0.0;
    for (i = 0; i < n; i++) {
        sum_sq += norm_weights[i] * norm_weights[i];
    }
    
    /* Avoid division by zero */
    if (sum_sq < 1e-300) {
        return 1.0;
    }
    
    return 1.0 / sum_sq;
}

/**
 * @brief Compute weighted mean for a 1D state component.
 * @param particles Pointer to particle array (cast to appropriate type by caller).
 * @param norm_weights Normalized weights.
 * @param n Number of particles.
 * @param state_size Size of each particle in bytes.
 * @param offset Byte offset of the component within the state.
 * @return Weighted mean.
 *
 * Note: This assumes the component at offset is a double.
 */
double fastpf_weighted_mean_component(const unsigned char* particles,
                                       const double* norm_weights,
                                       size_t n,
                                       size_t state_size,
                                       size_t offset)
{
    size_t i;
    double mean;
    const double* component;
    
    assert(particles != NULL);
    assert(norm_weights != NULL);
    assert(n > 0);
    
    mean = 0.0;
    for (i = 0; i < n; i++) {
        component = (const double*)(particles + i * state_size + offset);
        mean += norm_weights[i] * (*component);
    }
    
    return mean;
}

/**
 * @brief Compute weighted variance for a 1D state component.
 * @param particles Pointer to particle array.
 * @param norm_weights Normalized weights.
 * @param n Number of particles.
 * @param state_size Size of each particle in bytes.
 * @param offset Byte offset of the component within the state.
 * @param mean Pre-computed weighted mean (for efficiency).
 * @return Weighted variance.
 */
double fastpf_weighted_variance_component(const unsigned char* particles,
                                           const double* norm_weights,
                                           size_t n,
                                           size_t state_size,
                                           size_t offset,
                                           double mean)
{
    size_t i;
    double variance, diff;
    const double* component;
    
    assert(particles != NULL);
    assert(norm_weights != NULL);
    assert(n > 0);
    
    variance = 0.0;
    for (i = 0; i < n; i++) {
        component = (const double*)(particles + i * state_size + offset);
        diff = (*component) - mean;
        variance += norm_weights[i] * diff * diff;
    }
    
    return variance;
}
