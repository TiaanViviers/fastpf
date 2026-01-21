/**
 * @file test_rng.c
 * @brief Unit tests for RNG (PCG32).
 */

#include "fastpf.h"
#include "test_common.h"
#include <math.h>
#include <assert.h>

/* C90-compatible constants (avoid ULL suffix warnings) */
#define SEED_12345 12345
#define SEED_111   111
#define SEED_222   222
#define SEED_999   999
#define SEED_42    42
#define SEED_123   123

/* Simple erf() approximation (C90 doesn't guarantee erf in math.h) */
static double simple_erf(double x)
{
    /* Abramowitz and Stegun approximation, max error 1.5e-7 */
    double a1, a2, a3, a4, a5, p, t, y;
    int sign;
    
    sign = (x >= 0) ? 1 : -1;
    x = fabs(x);
    
    a1 = 0.254829592;
    a2 = -0.284496736;
    a3 = 1.421413741;
    a4 = -1.453152027;
    a5 = 1.061405429;
    p = 0.3275911;
    
    t = 1.0 / (1.0 + p * x);
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-x * x);
    
    return sign * y;
}

/* Normal CDF using erf: Φ(x) = 0.5 * (1 + erf(x/√2)) */
static double normal_cdf(double x)
{
    return 0.5 * (1.0 + simple_erf(x / 1.41421356237)); /* √2 ≈ 1.414... */
}

/* ========================================================================
 * Test: Deterministic output for fixed seed
 * ======================================================================== */
int test_rng_deterministic(void)
{
    fastpf_rng_t rng1, rng2;
    unsigned int u1, u2;
    int i;
    
    /* Same seed should produce same sequence */
    fastpf_rng_seed(&rng1, SEED_12345);
    fastpf_rng_seed(&rng2, SEED_12345);
    
    for (i = 0; i < 100; i++) {
        u1 = fastpf_rng_u32(&rng1);
        u2 = fastpf_rng_u32(&rng2);
        TEST_ASSERT(u1 == u2, "Deterministic test: same seed produces same output");
    }
    
    TEST_PASS("test_rng_deterministic");
    return 0;
}

/* ========================================================================
 * Test: Different seeds produce different sequences
 * ======================================================================== */
int test_rng_different_seeds(void)
{
    fastpf_rng_t rng1, rng2;
    unsigned int u1, u2;
    int different;
    int i;
    
    fastpf_rng_seed(&rng1, SEED_111);
    fastpf_rng_seed(&rng2, SEED_222);
    
    different = 0;
    for (i = 0; i < 10; i++) {
        u1 = fastpf_rng_u32(&rng1);
        u2 = fastpf_rng_u32(&rng2);
        if (u1 != u2) {
            different = 1;
            break;
        }
    }
    
    TEST_ASSERT(different, "Different seeds produce different sequences");
    TEST_PASS("test_rng_different_seeds");
    return 0;
}

/* ========================================================================
 * Test: Uniform01 in range (0, 1)
 * ======================================================================== */
int test_rng_uniform01_range(void)
{
    fastpf_rng_t rng;
    double u;
    int i;
    
    fastpf_rng_seed(&rng, SEED_999);
    
    for (i = 0; i < 1000; i++) {
        u = fastpf_rng_uniform01(&rng);
        TEST_ASSERT(u > 0.0 && u < 1.0, "Uniform01 in open interval (0,1)");
    }
    
    TEST_PASS("test_rng_uniform01_range");
    return 0;
}

/* ========================================================================
 * Test: Uniform01 empirical mean and variance
 * ======================================================================== */
int test_rng_uniform01_moments(void)
{
    fastpf_rng_t rng;
    double sum, sum_sq, mean, variance;
    double u;
    int i;
    int n;
    
    n = 10000;
    fastpf_rng_seed(&rng, SEED_42);
    
    sum = 0.0;
    sum_sq = 0.0;
    
    for (i = 0; i < n; i++) {
        u = fastpf_rng_uniform01(&rng);
        sum += u;
        sum_sq += u * u;
    }
    
    mean = sum / n;
    variance = (sum_sq / n) - (mean * mean);
    
    /* Uniform(0,1): E[X] = 0.5, Var[X] = 1/12 ≈ 0.0833 */
    TEST_ASSERT(fabs(mean - 0.5) < 0.01, "Uniform01 mean ≈ 0.5");
    TEST_ASSERT(fabs(variance - 1.0/12.0) < 0.01, "Uniform01 variance ≈ 1/12");
    
    TEST_PASS("test_rng_uniform01_moments");
    return 0;
}

/* ========================================================================
 * Test: Normal distribution moments
 * ======================================================================== */
int test_rng_normal_moments(void)
{
    fastpf_rng_t rng;
    double sum, sum_sq, mean, variance;
    double z;
    int i;
    int n;
    
    n = 10000;
    fastpf_rng_seed(&rng, SEED_123);
    
    sum = 0.0;
    sum_sq = 0.0;
    
    for (i = 0; i < n; i++) {
        z = fastpf_rng_normal(&rng);
        sum += z;
        sum_sq += z * z;
    }
    
    mean = sum / n;
    variance = (sum_sq / n) - (mean * mean);
    
    /* N(0,1): E[Z] = 0, Var[Z] = 1 */
    TEST_ASSERT(fabs(mean - 0.0) < 0.05, "Normal mean ≈ 0");
    TEST_ASSERT(fabs(variance - 1.0) < 0.05, "Normal variance ≈ 1");
    
    TEST_PASS("test_rng_normal_moments");
    return 0;
}

/* ========================================================================
 * Test: Normal distribution tail behavior
 * ======================================================================== */
int test_rng_normal_tails(void)
{
    fastpf_rng_t rng;
    double z, abs_z;
    int i, n;
    int count_gt3, count_gt4, count_gt5;
    double rate_gt3, rate_gt4;
    double expected_gt3, expected_gt4, expected_gt5;
    
    n = 1000000; /* 1 million samples for tail statistics */
    fastpf_rng_seed(&rng, SEED_42);
    
    count_gt3 = 0;
    count_gt4 = 0;
    count_gt5 = 0;
    
    for (i = 0; i < n; i++) {
        z = fastpf_rng_normal(&rng);
        abs_z = fabs(z);
        
        if (abs_z > 3.0) count_gt3++;
        if (abs_z > 4.0) count_gt4++;
        if (abs_z > 5.0) count_gt5++;
    }
    
    rate_gt3 = (double)count_gt3 / n;
    rate_gt4 = (double)count_gt4 / n;
    
    /* Expected rates for standard normal:
     * P(|Z| > 3) ≈ 0.0027 (0.27%) → ~2700 events in 1M
     * P(|Z| > 4) ≈ 6.3e-5 (0.0063%) → ~63 events in 1M
     * P(|Z| > 5) ≈ 5.7e-7 (0.000057%) → ~0.57 events in 1M (0 or 1 is normal)
     */
    expected_gt3 = 0.0027;
    expected_gt4 = 6.3e-5;
    expected_gt5 = 5.7e-7;
    
    /* Allow 50% relative error for P(|Z| > 3) and P(|Z| > 4) */
    TEST_ASSERT(fabs(rate_gt3 - expected_gt3) < 0.5 * expected_gt3,
                "Normal tail: P(|Z| > 3) within 50% of expected");
    TEST_ASSERT(fabs(rate_gt4 - expected_gt4) < 0.5 * expected_gt4,
                "Normal tail: P(|Z| > 4) within 50% of expected");
    
    /* For |Z| > 5, expect 0-1 events in 1M samples, allow up to 10 (very loose) */
    TEST_ASSERT(count_gt5 <= 10,
                "Normal tail: P(|Z| > 5) not impossibly high");
    
    /* Avoid unused variable warning */
    (void)expected_gt5;
    
    TEST_PASS("test_rng_normal_tails");
    return 0;
}

/* ========================================================================
 * Test: Normal distribution symmetry
 * ======================================================================== */
int test_rng_normal_symmetry(void)
{
    fastpf_rng_t rng;
    double z, sum;
    int i, n;
    int count_positive;
    double proportion_positive;
    double mean;
    
    n = 100000;
    fastpf_rng_seed(&rng, SEED_999);
    
    count_positive = 0;
    sum = 0.0;
    
    for (i = 0; i < n; i++) {
        z = fastpf_rng_normal(&rng);
        if (z > 0.0) {
            count_positive++;
        }
        sum += z;
    }
    
    proportion_positive = (double)count_positive / n;
    mean = sum / n;
    
    /* Check symmetry: ~50% positive, mean ~0 */
    TEST_ASSERT(fabs(proportion_positive - 0.5) < 0.01,
                "Normal symmetry: ~50% positive values");
    TEST_ASSERT(fabs(mean) < 0.01,
                "Normal symmetry: mean ≈ 0");
    
    TEST_PASS("test_rng_normal_symmetry");
    return 0;
}

/* ========================================================================
 * Test: Normal distribution histogram (KS-style bin check)
 * ======================================================================== */
int test_rng_normal_histogram(void)
{
    fastpf_rng_t rng;
    double z;
    int i, n, bin;
    int hist[9]; /* Bins: (-inf,-3], (-3,-2], ..., (2,3], (3,inf) */
    double bin_edges[8]; /* -3, -2, -1, 0, 1, 2, 3 */
    double expected_prob[9];
    double observed_prob[9];
    double rel_error;
    
    n = 200000;
    fastpf_rng_seed(&rng, SEED_42);
    
    /* Initialize histogram */
    for (i = 0; i < 9; i++) {
        hist[i] = 0;
    }
    
    /* Bin edges */
    for (i = 0; i < 8; i++) {
        bin_edges[i] = -3.0 + i; /* -3, -2, -1, 0, 1, 2, 3 */
    }
    
    /* Collect samples into histogram */
    for (i = 0; i < n; i++) {
        z = fastpf_rng_normal(&rng);
        
        /* Find bin */
        if (z <= -3.0) {
            bin = 0;
        } else if (z > 3.0) {
            bin = 8;
        } else {
            /* Bins 1-7 for [-3,-2], [-2,-1], ..., [2,3] */
            bin = (int)(z + 3.0) + 1;
            if (bin < 1) bin = 1;
            if (bin > 7) bin = 7;
        }
        
        hist[bin]++;
    }
    
    /* Compute expected probabilities using normal CDF */
    expected_prob[0] = normal_cdf(-3.0); /* P(Z <= -3) */
    for (i = 1; i < 8; i++) {
        expected_prob[i] = normal_cdf(bin_edges[i]) - normal_cdf(bin_edges[i-1]);
    }
    expected_prob[8] = 1.0 - normal_cdf(3.0); /* P(Z > 3) */
    
    /* Check observed vs expected (allow 25% relative error for mid bins, absolute error for tails) */
    for (i = 0; i < 9; i++) {
        observed_prob[i] = (double)hist[i] / n;
        
        if (expected_prob[i] > 0.10) {
            /* Large bins (>10% probability): allow 15% relative error */
            rel_error = fabs(observed_prob[i] - expected_prob[i]) / expected_prob[i];
            TEST_ASSERT(rel_error < 0.15,
                        "Normal histogram: large bins within 15% of expected");
        } else if (expected_prob[i] > 0.01) {
            /* Mid bins (1%-10%): allow 30% relative error */
            rel_error = fabs(observed_prob[i] - expected_prob[i]) / expected_prob[i];
            TEST_ASSERT(rel_error < 0.30,
                        "Normal histogram: mid bins within 30% of expected");
        } else {
            /* Tail bins (<1%): use loose absolute error (catches gross errors only) */
            TEST_ASSERT(fabs(observed_prob[i] - expected_prob[i]) < 0.01,
                        "Normal histogram: tail bins reasonable");
        }
    }
    
    TEST_PASS("test_rng_normal_histogram");
    return 0;
}

/* ========================================================================
 * Test: Independence (lag-1 autocorrelation should be near 0)
 * ======================================================================== */
int test_rng_normal_independence(void)
{
    fastpf_rng_t rng;
    double z, z_prev;
    double sum_z, sum_z_prev, sum_z_z_prev, sum_z_sq, sum_z_prev_sq;
    double mean_z, mean_z_prev, cov, var_z, var_z_prev, rho;
    int i, n;
    
    n = 100000;
    fastpf_rng_seed(&rng, SEED_123);
    
    sum_z = 0.0;
    sum_z_prev = 0.0;
    sum_z_z_prev = 0.0;
    sum_z_sq = 0.0;
    sum_z_prev_sq = 0.0;
    
    /* First sample */
    z_prev = fastpf_rng_normal(&rng);
    
    /* Collect lag-1 pairs */
    for (i = 1; i < n; i++) {
        z = fastpf_rng_normal(&rng);
        
        sum_z += z;
        sum_z_prev += z_prev;
        sum_z_z_prev += z * z_prev;
        sum_z_sq += z * z;
        sum_z_prev_sq += z_prev * z_prev;
        
        z_prev = z;
    }
    
    /* Compute lag-1 autocorrelation */
    mean_z = sum_z / (n - 1);
    mean_z_prev = sum_z_prev / (n - 1);
    
    cov = (sum_z_z_prev / (n - 1)) - (mean_z * mean_z_prev);
    var_z = (sum_z_sq / (n - 1)) - (mean_z * mean_z);
    var_z_prev = (sum_z_prev_sq / (n - 1)) - (mean_z_prev * mean_z_prev);
    
    rho = cov / sqrt(var_z * var_z_prev);
    
    /* Lag-1 correlation should be near 0 for independent samples */
    TEST_ASSERT(fabs(rho) < 0.01,
                "Normal independence: |rho1| < 0.01");
    
    TEST_PASS("test_rng_normal_independence");
    return 0;
}

/* ========================================================================
 * Main test runner
 * ======================================================================== */
int main(void)
{
    int result = 0;
    
    printf("=== RNG Unit Tests ===\n");
    
    result |= test_rng_deterministic();
    result |= test_rng_different_seeds();
    result |= test_rng_uniform01_range();
    result |= test_rng_uniform01_moments();
    result |= test_rng_normal_moments();
    result |= test_rng_normal_tails();
    result |= test_rng_normal_symmetry();
    result |= test_rng_normal_histogram();
    result |= test_rng_normal_independence();
    
    if (result == 0) {
        printf("\n=== All RNG tests passed ===\n");
    } else {
        printf("\n=== Some RNG tests failed ===\n");
    }
    
    return result;
}
