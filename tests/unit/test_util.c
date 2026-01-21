/**
 * @file test_util.c
 * @brief Unit tests for utility functions (log-sum-exp, etc.).
 */

#include "fastpf.h"
#include "test_common.h"
#include <math.h>
#include <float.h>

/* Forward declaration of internal function */
extern double fastpf_logsumexp(const double* log_weights, size_t n);
extern double fastpf_normalize_weights(double* log_weights, double* norm_weights, size_t n);

/* ========================================================================
 * Test: logsumexp with normal values
 * ======================================================================== */
int test_logsumexp_normal(void)
{
    double log_weights[3];
    double result, expected;
    
    /* log(1) + log(2) + log(3) = log(1 + 2 + 3) = log(6) */
    log_weights[0] = log(1.0);
    log_weights[1] = log(2.0);
    log_weights[2] = log(3.0);
    
    result = fastpf_logsumexp(log_weights, 3);
    expected = log(6.0);
    
    TEST_ASSERT(fabs(result - expected) < 1e-10, "logsumexp([log(1), log(2), log(3)]) = log(6)");
    
    TEST_PASS("test_logsumexp_normal");
    return 0;
}

/* ========================================================================
 * Test: logsumexp with extreme values (stability)
 * ======================================================================== */
int test_logsumexp_extreme(void)
{
    double log_weights[4];
    double result, expected;
    
    /* Large positive values */
    log_weights[0] = 1000.0;
    log_weights[1] = 1001.0;
    log_weights[2] = 1002.0;
    log_weights[3] = 999.0;
    
    result = fastpf_logsumexp(log_weights, 4);
    
    /* Result should be close to max + log(sum(exp(x - max))) */
    expected = 1002.0 + log(exp(1000.0 - 1002.0) + exp(1001.0 - 1002.0) + 
                             exp(1002.0 - 1002.0) + exp(999.0 - 1002.0));
    
    TEST_ASSERT(fabs(result - expected) < 1e-8, "logsumexp stable with large values");
    TEST_ASSERT(!isinf(result) && !isnan(result), "logsumexp result is finite");
    
    TEST_PASS("test_logsumexp_extreme");
    return 0;
}

/* ========================================================================
 * Test: logsumexp with -inf values
 * ======================================================================== */
int test_logsumexp_neginf(void)
{
    double log_weights[4];
    double result, expected;
    
    /* Mix of -inf and finite values */
    log_weights[0] = -INFINITY;
    log_weights[1] = log(2.0);
    log_weights[2] = -INFINITY;
    log_weights[3] = log(3.0);
    
    result = fastpf_logsumexp(log_weights, 4);
    expected = log(5.0); /* log(2 + 3) */
    
    TEST_ASSERT(fabs(result - expected) < 1e-10, "logsumexp handles -inf correctly");
    
    TEST_PASS("test_logsumexp_neginf");
    return 0;
}

/* ========================================================================
 * Test: normalize_weights produces normalized weights
 * ======================================================================== */
int test_normalize_weights(void)
{
    double log_weights[4];
    double norm_weights[4];
    double sum;
    int i;
    
    /* Arbitrary log-weights */
    log_weights[0] = log(1.0);
    log_weights[1] = log(4.0);
    log_weights[2] = log(2.0);
    log_weights[3] = log(3.0);
    
    fastpf_normalize_weights(log_weights, norm_weights, 4);
    
    /* Check that normalized weights sum to 1 */
    sum = 0.0;
    for (i = 0; i < 4; i++) {
        sum += norm_weights[i];
    }
    
    TEST_ASSERT(fabs(sum - 1.0) < 1e-10, "Normalized weights sum to 1");
    
    /* Check specific values: w[i] = exp(log_weights[i]) / sum */
    TEST_ASSERT(fabs(norm_weights[0] - 0.1) < 1e-10, "norm_weights[0] = 1/10");
    TEST_ASSERT(fabs(norm_weights[1] - 0.4) < 1e-10, "norm_weights[1] = 4/10");
    TEST_ASSERT(fabs(norm_weights[2] - 0.2) < 1e-10, "norm_weights[2] = 2/10");
    TEST_ASSERT(fabs(norm_weights[3] - 0.3) < 1e-10, "norm_weights[3] = 3/10");
    
    TEST_PASS("test_normalize_weights");
    return 0;
}

/* ========================================================================
 * Main test runner
 * ======================================================================== */
int main(void)
{
    int result = 0;
    
    printf("=== Utility Function Unit Tests ===\n");
    
    result |= test_logsumexp_normal();
    result |= test_logsumexp_extreme();
    result |= test_logsumexp_neginf();
    result |= test_normalize_weights();
    
    if (result == 0) {
        printf("\n=== All utility tests passed ===\n");
    } else {
        printf("\n=== Some utility tests failed ===\n");
    }
    
    return result;
}
