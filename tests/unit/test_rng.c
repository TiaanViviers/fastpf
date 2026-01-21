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
    
    if (result == 0) {
        printf("\n=== All RNG tests passed ===\n");
    } else {
        printf("\n=== Some RNG tests failed ===\n");
    }
    
    return result;
}
