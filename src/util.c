/**
 * @file util.c
 * @brief Utility functions: log-sum-exp, safe math, error handling.
 */

#include "fastpf.h"
#include "fastpf_internal.h"
#include <math.h>
#include <float.h>
#include <assert.h>

/* C90-compatible infinity */
#ifndef INFINITY
    #define INFINITY (1.0 / 0.0)
#endif

/* C90-compatible isnan/isinf */
#ifndef isnan
    #define isnan(x) ((x) != (x))
#endif
#ifndef isinf
    #define isinf(x) (!isnan(x) && isnan((x) - (x)))
#endif

/* ========================================================================
 * Log-sum-exp for numerical stability
 * ======================================================================== */

/**
 * @brief Compute log(sum(exp(log_weights))) numerically stably.
 * @param log_weights Array of log-weights.
 * @param n Number of elements.
 * @return log(sum(exp(log_weights[i]))).
 */
double fastpf_logsumexp(const double* log_weights, size_t n)
{
    size_t i;
    double max_log_weight, sum_exp, result;
    
    assert(log_weights != NULL);
    assert(n > 0);
    
    /* Find maximum log-weight */
    max_log_weight = log_weights[0];
    for (i = 1; i < n; i++) {
        if (log_weights[i] > max_log_weight) {
            max_log_weight = log_weights[i];
        }
    }
    
    /* Handle edge case: all weights are -inf */
    if (max_log_weight == -INFINITY) {
        return -INFINITY;
    }
    
    /* Compute sum(exp(log_weights - max)) */
    sum_exp = 0.0;
    for (i = 0; i < n; i++) {
        sum_exp += exp(log_weights[i] - max_log_weight);
    }
    
    result = max_log_weight + log(sum_exp);
    
    /* Debug: check for NaN */
    assert(!isnan(result));
    
    return result;
}

/**
 * @brief Normalize log-weights in-place and populate normalized weights.
 * @param log_weights Input: array of log-weights (modified in-place to normalized log-weights).
 * @param norm_weights Output: array of normalized weights in linear space.
 * @param n Number of elements.
 * @return Log of the sum of unnormalized weights (useful for log-likelihood tracking).
 */
double fastpf_normalize_weights(double* log_weights, double* norm_weights, size_t n)
{
    size_t i;
    double log_sum, sum_check;
    
    assert(log_weights != NULL);
    assert(norm_weights != NULL);
    assert(n > 0);
    
    /* Compute log(sum(exp(log_weights))) */
    log_sum = fastpf_logsumexp(log_weights, n);
    
    /* Handle edge case: all particles have zero likelihood (log_sum = -INFINITY).
     * This can happen with outlier observations. Set weights to uniform. */
    if (log_sum == -INFINITY) {
        for (i = 0; i < n; i++) {
            log_weights[i] = -log((double)n);
            norm_weights[i] = 1.0 / (double)n;
        }
        return -INFINITY;
    }
    
    /* Normalize: log_weights[i] -= log_sum, norm_weights[i] = exp(log_weights[i]) */
    sum_check = 0.0;
    for (i = 0; i < n; i++) {
        log_weights[i] -= log_sum;
        norm_weights[i] = exp(log_weights[i]);
        sum_check += norm_weights[i];
        
        /* Debug: no NaNs */
        assert(!isnan(log_weights[i]));
        assert(!isnan(norm_weights[i]));
    }
    
    /* Debug: normalized weights should sum to ~1 */
    assert(fabs(sum_check - 1.0) < 1e-6);
    
    return log_sum;
}

/**
 * @brief Check if a double value is finite (not NaN or Inf).
 * @param x Value to check.
 * @return 1 if finite, 0 otherwise.
 */
int fastpf_isfinite(double x)
{
    return !isnan(x) && !isinf(x);
}

/**
 * @brief Compute maximum element in array.
 * @param arr Array of doubles.
 * @param n Number of elements.
 * @return Maximum value.
 */
double fastpf_max(const double* arr, size_t n)
{
    size_t i;
    double max_val;
    
    assert(arr != NULL);
    assert(n > 0);
    
    max_val = arr[0];
    for (i = 1; i < n; i++) {
        if (arr[i] > max_val) {
            max_val = arr[i];
        }
    }
    
    return max_val;
}
