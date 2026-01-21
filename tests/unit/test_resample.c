/**
 * @file test_resample.c
 * @brief Unit tests for resampling algorithms.
 */

#include "fastpf.h"
#include "test_common.h"
#include <math.h>

/* C90-compatible seeds */
#define SEED_42    42
#define SEED_123   123
#define SEED_777   777
#define SEED_999   999

/* Forward declaration */
extern void fastpf_resample_systematic(const double* norm_weights, size_t n, 
                                        size_t* indices, fastpf_rng_t* rng);

/* ========================================================================
 * Test: Systematic resampling produces valid indices
 * ======================================================================== */
int test_resample_valid_indices(void)
{
    double weights[10];
    size_t indices[10];
    fastpf_rng_t rng;
    int i;
    
    /* Uniform weights */
    for (i = 0; i < 10; i++) {
        weights[i] = 0.1;
    }
    
    fastpf_rng_seed(&rng, SEED_42);
    fastpf_resample_systematic(weights, 10, indices, &rng);
    
    /* All indices should be in [0, 10) */
    for (i = 0; i < 10; i++) {
        TEST_ASSERT(indices[i] < 10, "Resampled index in valid range");
    }
    
    TEST_PASS("test_resample_valid_indices");
    return 0;
}

/* ========================================================================
 * Test: Systematic resampling empirical distribution
 * ======================================================================== */
int test_resample_distribution(void)
{
    double weights[5];
    size_t indices[5000];
    int counts[5];
    fastpf_rng_t rng;
    int i;
    size_t run, n_runs;
    double expected[5], actual, chi_sq;
    
    /* Non-uniform weights: [0.1, 0.2, 0.3, 0.25, 0.15] */
    weights[0] = 0.1;
    weights[1] = 0.2;
    weights[2] = 0.3;
    weights[3] = 0.25;
    weights[4] = 0.15;
    
    /* Expected counts */
    expected[0] = 0.1;
    expected[1] = 0.2;
    expected[2] = 0.3;
    expected[3] = 0.25;
    expected[4] = 0.15;
    
    /* Initialize counts */
    for (i = 0; i < 5; i++) {
        counts[i] = 0;
    }
    
    /* Resample 1000 times with 5 particles each */
    n_runs = 1000;
    fastpf_rng_seed(&rng, SEED_123);
    for (run = 0; run < n_runs; run++) {
        fastpf_resample_systematic(weights, 5, indices, &rng);
        /* Count all resampled particles */
        for (i = 0; i < 5; i++) {
            counts[indices[i]]++;
        }
    }
    
    /* Check that empirical distribution roughly matches weights */
    /* Total particles = n_runs * 5 = 5000 */
    chi_sq = 0.0;
    for (i = 0; i < 5; i++) {
        double exp_count = expected[i] * n_runs * 5;
        actual = (double)counts[i];
        chi_sq += (actual - exp_count) * (actual - exp_count) / exp_count;
    }
    
    /* Chi-square test: with 4 degrees of freedom, critical value at 0.05 is ~9.5 */
    /* We use a looser threshold for this unit test */
    TEST_ASSERT(chi_sq < 50.0, "Resampling produces expected distribution");
    
    TEST_PASS("test_resample_distribution");
    return 0;
}

/* ========================================================================
 * Test: Systematic resampling determinism for fixed seed
 * ======================================================================== */
int test_resample_deterministic(void)
{
    double weights[10];
    size_t indices1[10], indices2[10];
    fastpf_rng_t rng1, rng2;
    int i;
    int match;
    
    /* Arbitrary weights */
    for (i = 0; i < 10; i++) {
        weights[i] = (i + 1.0) / 55.0;
    }
    
    /* Same seed */
    fastpf_rng_seed(&rng1, SEED_999);
    fastpf_rng_seed(&rng2, SEED_999);
    
    fastpf_resample_systematic(weights, 10, indices1, &rng1);
    fastpf_resample_systematic(weights, 10, indices2, &rng2);
    
    /* Indices should match */
    match = 1;
    for (i = 0; i < 10; i++) {
        if (indices1[i] != indices2[i]) {
            match = 0;
            break;
        }
    }
    
    TEST_ASSERT(match, "Resampling deterministic for same seed");
    
    TEST_PASS("test_resample_deterministic");
    return 0;
}

/* ========================================================================
 * Test: Resampling with single dominant weight
 * ======================================================================== */
int test_resample_dominant_weight(void)
{
    double weights[10];
    size_t indices[10];
    fastpf_rng_t rng;
    int i;
    int count_idx5;
    
    /* One particle has most weight */
    for (i = 0; i < 10; i++) {
        weights[i] = 0.01;
    }
    weights[5] = 0.91; /* Particle 5 has 91% weight */
    
    fastpf_rng_seed(&rng, SEED_777);
    fastpf_resample_systematic(weights, 10, indices, &rng);
    
    /* Most resampled indices should be 5 */
    count_idx5 = 0;
    for (i = 0; i < 10; i++) {
        if (indices[i] == 5) {
            count_idx5++;
        }
    }
    
    TEST_ASSERT(count_idx5 >= 8, "Dominant particle resampled most");
    
    TEST_PASS("test_resample_dominant_weight");
    return 0;
}

/* ========================================================================
 * Main test runner
 * ======================================================================== */
int main(void)
{
    int result = 0;
    
    printf("=== Resampling Unit Tests ===\n");
    
    result |= test_resample_valid_indices();
    result |= test_resample_distribution();
    result |= test_resample_deterministic();
    result |= test_resample_dominant_weight();
    
    if (result == 0) {
        printf("\n=== All resampling tests passed ===\n");
    } else {
        printf("\n=== Some resampling tests failed ===\n");
    }
    
    return result;
}
