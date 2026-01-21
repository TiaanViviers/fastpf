/**
 * @file test_stats.c
 * @brief Unit tests for statistics functions (ESS, weighted moments).
 */

#include "fastpf.h"
#include "test_common.h"
#include <math.h>

/* Forward declaration */
extern double fastpf_compute_ess(const double* norm_weights, size_t n);

/* ========================================================================
 * Test: ESS with uniform weights
 * ======================================================================== */
int test_ess_uniform(void)
{
    double weights[100];
    double ess;
    int i;
    
    /* Uniform weights: w[i] = 1/N */
    for (i = 0; i < 100; i++) {
        weights[i] = 1.0 / 100.0;
    }
    
    ess = fastpf_compute_ess(weights, 100);
    
    /* ESS = 1 / sum(w^2) = 1 / (N * (1/N)^2) = N */
    TEST_ASSERT(fabs(ess - 100.0) < 1e-6, "ESS with uniform weights = N");
    
    TEST_PASS("test_ess_uniform");
    return 0;
}

/* ========================================================================
 * Test: ESS with single dominant weight
 * ======================================================================== */
int test_ess_single_weight(void)
{
    double weights[100];
    double ess;
    int i;
    
    /* One weight dominates: w[0] ≈ 1, rest ≈ 0 */
    weights[0] = 0.99;
    for (i = 1; i < 100; i++) {
        weights[i] = 0.01 / 99.0;
    }
    
    ess = fastpf_compute_ess(weights, 100);
    
    /* ESS ≈ 1 / (0.99^2 + 99*(0.01/99)^2) ≈ 1.02 */
    TEST_ASSERT(ess >= 1.0 && ess < 2.0, "ESS with single dominant weight ≈ 1");
    
    TEST_PASS("test_ess_single_weight");
    return 0;
}

/* ========================================================================
 * Test: ESS with half-half weights
 * ======================================================================== */
int test_ess_half_half(void)
{
    double weights[100];
    double ess;
    int i;
    
    /* Half particles with weight 2/N, half with 0 */
    for (i = 0; i < 50; i++) {
        weights[i] = 2.0 / 100.0;
    }
    for (i = 50; i < 100; i++) {
        weights[i] = 0.0;
    }
    
    ess = fastpf_compute_ess(weights, 100);
    
    /* ESS = 1 / sum(w^2) = 1 / (50 * (2/100)^2) = 50 */
    TEST_ASSERT(fabs(ess - 50.0) < 1e-6, "ESS with half particles having weight");
    
    TEST_PASS("test_ess_half_half");
    return 0;
}

/* ========================================================================
 * Test: ESS bounds check
 * ======================================================================== */
int test_ess_bounds(void)
{
    double weights[100];
    double ess;
    int i;
    
    /* Arbitrary non-uniform weights */
    for (i = 0; i < 100; i++) {
        weights[i] = (i + 1.0) / 5050.0; /* sum = 1 */
    }
    
    ess = fastpf_compute_ess(weights, 100);
    
    /* ESS should be in [1, N] */
    TEST_ASSERT(ess >= 1.0 && ess <= 100.0, "ESS in valid range [1, N]");
    
    TEST_PASS("test_ess_bounds");
    return 0;
}

/* ========================================================================
 * Main test runner
 * ======================================================================== */
int main(void)
{
    int result = 0;
    
    printf("=== Statistics Unit Tests ===\n");
    
    result |= test_ess_uniform();
    result |= test_ess_single_weight();
    result |= test_ess_half_half();
    result |= test_ess_bounds();
    
    if (result == 0) {
        printf("\n=== All statistics tests passed ===\n");
    } else {
        printf("\n=== Some statistics tests failed ===\n");
    }
    
    return result;
}
