/**
 * @file test_checkpoint.c
 * @brief Comprehensive checkpoint system validation.
 *
 * Test Strategy:
 * --------------
 * 1. Deterministic resume: Prove restore is bitwise identical
 * 2. Failure modes: Verify all validation checks catch corruption
 * 3. Pattern testing: Both init+load and validate+overwrite workflows
 * 4. Edge cases: Empty particles, extreme weights, etc.
 */

#include "fastpf.h"
#include "../../src/fastpf_internal.h"  /* For direct state access in tests */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

/* Test framework macros */
#define TEST_ASSERT(cond, msg) do { \
    if (!(cond)) { \
        printf("FAIL: %s\n", msg); \
        return 0; \
    } \
} while(0)

#define SEED_42 42

/* ========================================================================
 * Simple test model: 1D random walk
 * ======================================================================== */

typedef struct {
    double process_noise;  /* std dev of state transition */
    double meas_noise;     /* std dev of measurement */
} rw_model_ctx_t;

void rw_prior_sample(void* ctx, void* x0_out, fastpf_rng_t* rng)
{
    double* x = (double*)x0_out;
    (void)ctx;
    *x = fastpf_rng_normal(rng);  /* N(0,1) prior */
}

void rw_transition_sample(void* ctx, const void* x_prev, void* x_out, fastpf_rng_t* rng)
{
    const rw_model_ctx_t* model = (const rw_model_ctx_t*)ctx;
    const double* x_old = (const double*)x_prev;
    double* x_new = (double*)x_out;
    *x_new = *x_old + model->process_noise * fastpf_rng_normal(rng);
}

double rw_log_likelihood(void* ctx, const void* x, const void* y)
{
    const rw_model_ctx_t* model = (const rw_model_ctx_t*)ctx;
    const double* x_val = (const double*)x;
    const double* y_val = (const double*)y;
    double diff = *y_val - *x_val;
    double var = model->meas_noise * model->meas_noise;
    return -0.5 * (diff * diff / var + log(2.0 * 3.14159265358979323846 * var));
}

/* ========================================================================
 * FNV-1a hash for determinism validation
 * ======================================================================== */

#define FNV_OFFSET_BASIS 0xcbf29ce484222325ULL
#define FNV_PRIME 0x100000001b3ULL

uint64_t fnv1a_hash(const void* data, size_t len)
{
    const unsigned char* bytes = (const unsigned char*)data;
    uint64_t hash = FNV_OFFSET_BASIS;
    size_t i;
    
    for (i = 0; i < len; i++) {
        hash ^= (uint64_t)bytes[i];
        hash *= FNV_PRIME;
    }
    
    return hash;
}

/**
 * @brief Hash complete PF state for determinism checking.
 *
 * Hashes particles, log-weights, and RNG state. This is the "fingerprint"
 * that must match between checkpointed and non-checkpointed runs.
 */
uint64_t hash_pf_state(const fastpf_t* pf)
{
    uint64_t hash = FNV_OFFSET_BASIS;
    size_t particles_bytes, weights_bytes;
    
    /* Hash particles */
    particles_bytes = pf->cfg.n_particles * pf->cfg.state_size;
    hash ^= fnv1a_hash(pf->particles_curr, particles_bytes);
    hash *= FNV_PRIME;
    
    /* Hash log-weights */
    weights_bytes = pf->cfg.n_particles * sizeof(double);
    hash ^= fnv1a_hash(pf->log_weights, weights_bytes);
    hash *= FNV_PRIME;
    
    /* Hash RNG state */
    hash ^= fnv1a_hash(&pf->rng.state, sizeof(uint64_t));
    hash *= FNV_PRIME;
    hash ^= fnv1a_hash(&pf->rng.inc, sizeof(uint64_t));
    hash *= FNV_PRIME;
    
    return hash;
}

/* ========================================================================
 * Test 1: Deterministic resume (the critical test)
 * ======================================================================== */

/**
 * @brief Core determinism test: checkpoint + resume == never stopped.
 *
 * Strategy:
 * ---------
 * Run 1 (reference): K steps, record hash, continue L more steps, record final hash
 * Run 2 (checkpoint): K steps, checkpoint, restore, continue L more steps
 *
 * Assertions:
 * -----------
 * 1. Hash after K steps matches between runs (checkpoint is correct snapshot)
 * 2. Final hash after K+L steps matches (deterministic continuation)
 * 3. Resampling decisions match (diag.resampled identical at each step)
 *
 * This test catches:
 * - Missing state in checkpoint (hash would diverge)
 * - RNG corruption (trajectory would differ)
 * - Weight corruption (resampling decisions would differ)
 */
int test_checkpoint_deterministic_resume(void)
{
    rw_model_ctx_t model_ctx;
    fastpf_model_t model;
    fastpf_cfg_t cfg;
    fastpf_t pf1, pf2;
    double obs[20];  /* Observations for K+L steps */
    void* checkpoint_blob;
    size_t checkpoint_size;
    int i, k, result;
    uint64_t hash1_k, hash1_final, hash2_k, hash2_final;
    int resample_seq1[20], resample_seq2[20];  /* Track resampling decisions */
    
    /* Setup model */
    model_ctx.process_noise = 0.1;
    model_ctx.meas_noise = 1.0;
    model.ctx = &model_ctx;
    model.prior_sample = rw_prior_sample;
    model.transition_sample = rw_transition_sample;
    model.log_likelihood = rw_log_likelihood;
    model.rejuvenate = NULL;
    
    /* Setup config (small N for speed) */
    fastpf_cfg_init(&cfg, 100, sizeof(double));
    cfg.rng_seed = SEED_42;
    cfg.resample_threshold = 0.5;
    cfg.num_threads = -1;  /* Force serial for test */
    
    /* Generate fixed observation sequence */
    {
        fastpf_rng_t obs_rng;
        fastpf_rng_seed(&obs_rng, SEED_42 + 1000);
        for (i = 0; i < 20; i++) {
            obs[i] = fastpf_rng_normal(&obs_rng);
        }
    }
    
    /* -----------------------------------------------------------------------
     * Run 1: Reference run (no checkpoint)
     * ----------------------------------------------------------------------- */
    result = fastpf_init(&pf1, &cfg, &model);
    TEST_ASSERT(result == FASTPF_SUCCESS, "Reference PF init failed");
    
    /* Run K=10 steps */
    k = 10;
    for (i = 0; i < k; i++) {
        result = fastpf_step(&pf1, &obs[i]);
        TEST_ASSERT(result == FASTPF_SUCCESS, "Reference step failed");
        resample_seq1[i] = fastpf_get_diagnostics(&pf1)->resampled;
    }
    hash1_k = hash_pf_state(&pf1);
    
    /* Continue for L=10 more steps */
    for (i = k; i < 20; i++) {
        result = fastpf_step(&pf1, &obs[i]);
        TEST_ASSERT(result == FASTPF_SUCCESS, "Reference step failed");
        resample_seq1[i] = fastpf_get_diagnostics(&pf1)->resampled;
    }
    hash1_final = hash_pf_state(&pf1);
    
    /* -----------------------------------------------------------------------
     * Run 2: Checkpoint at K, restore, continue
     * ----------------------------------------------------------------------- */
    result = fastpf_init(&pf2, &cfg, &model);
    TEST_ASSERT(result == FASTPF_SUCCESS, "Checkpoint PF init failed");
    
    /* Run K=10 steps */
    for (i = 0; i < k; i++) {
        result = fastpf_step(&pf2, &obs[i]);
        TEST_ASSERT(result == FASTPF_SUCCESS, "Checkpoint step failed");
        resample_seq2[i] = fastpf_get_diagnostics(&pf2)->resampled;
    }
    hash2_k = hash_pf_state(&pf2);
    
    /* Hash after K steps must match */
    TEST_ASSERT(hash1_k == hash2_k,
                "Determinism: hashes differ before checkpoint");
    
    /* Save checkpoint */
    checkpoint_size = fastpf_checkpoint_bytes(&pf2);
    TEST_ASSERT(checkpoint_size > 0, "Checkpoint size is zero");
    
    checkpoint_blob = malloc(checkpoint_size);
    TEST_ASSERT(checkpoint_blob != NULL, "Checkpoint allocation failed");
    
    result = fastpf_checkpoint_write(&pf2, checkpoint_blob, checkpoint_size);
    TEST_ASSERT(result == FASTPF_SUCCESS, "Checkpoint write failed");
    
    /* Free and recreate PF2, then restore */
    fastpf_free(&pf2);
    
    memset(&pf2, 0, sizeof(pf2));
    pf2.model = model;  /* Must set callbacks before restore */
    
    result = fastpf_checkpoint_read(&pf2, &cfg, checkpoint_blob, checkpoint_size);
    TEST_ASSERT(result == FASTPF_SUCCESS, "Checkpoint restore failed");
    
    free(checkpoint_blob);
    
    /* Continue for L=10 more steps after restore */
    for (i = k; i < 20; i++) {
        result = fastpf_step(&pf2, &obs[i]);
        TEST_ASSERT(result == FASTPF_SUCCESS, "Post-restore step failed");
        resample_seq2[i] = fastpf_get_diagnostics(&pf2)->resampled;
    }
    hash2_final = hash_pf_state(&pf2);
    
    /* Final hashes must match (determinism guarantee) */
    TEST_ASSERT(hash1_final == hash2_final,
                "Determinism: final hashes differ after restore");
    
    /* Resampling decisions must match at every step */
    for (i = 0; i < 20; i++) {
        if (resample_seq1[i] != resample_seq2[i]) {
            printf("FAIL: Resampling mismatch at step %d (run1=%d, run2=%d)\n",
                   i, resample_seq1[i], resample_seq2[i]);
            return 0;
        }
    }
    
    /* Cleanup */
    fastpf_free(&pf1);
    fastpf_free(&pf2);
    
    return 1;
}

/* ========================================================================
 * Test 2: Pattern 1 (load into uninitialized PF)
 * ======================================================================== */

int test_checkpoint_pattern1_load_and_init(void)
{
    rw_model_ctx_t model_ctx;
    fastpf_model_t model;
    fastpf_cfg_t cfg;
    fastpf_t pf1, pf2;
    void* checkpoint_blob;
    size_t checkpoint_size;
    int result;
    
    /* Setup model */
    model_ctx.process_noise = 0.1;
    model_ctx.meas_noise = 1.0;
    model.ctx = &model_ctx;
    model.prior_sample = rw_prior_sample;
    model.transition_sample = rw_transition_sample;
    model.log_likelihood = rw_log_likelihood;
    model.rejuvenate = NULL;
    
    /* Create and checkpoint a PF */
    fastpf_cfg_init(&cfg, 50, sizeof(double));
    cfg.rng_seed = SEED_42;
    
    result = fastpf_init(&pf1, &cfg, &model);
    TEST_ASSERT(result == FASTPF_SUCCESS, "PF1 init failed");
    
    checkpoint_size = fastpf_checkpoint_bytes(&pf1);
    checkpoint_blob = malloc(checkpoint_size);
    TEST_ASSERT(checkpoint_blob != NULL, "Allocation failed");
    
    result = fastpf_checkpoint_write(&pf1, checkpoint_blob, checkpoint_size);
    TEST_ASSERT(result == FASTPF_SUCCESS, "Checkpoint write failed");
    
    /* Pattern 1: Load into uninitialized PF */
    memset(&pf2, 0, sizeof(pf2));
    pf2.model = model;  /* Must set callbacks */
    
    result = fastpf_checkpoint_read(&pf2, &cfg, checkpoint_blob, checkpoint_size);
    TEST_ASSERT(result == FASTPF_SUCCESS,
                "Pattern 1: checkpoint_read failed");
    TEST_ASSERT(pf2.initialized == 1,
                "Pattern 1: PF not marked initialized");
    TEST_ASSERT(pf2.particles_curr != NULL,
                "Pattern 1: particles not allocated");
    TEST_ASSERT(pf2.log_weights != NULL,
                "Pattern 1: weights not allocated");
    
    /* Verify N and state_size match */
    TEST_ASSERT(pf2.cfg.n_particles == 50,
                "Pattern 1: n_particles mismatch");
    TEST_ASSERT(pf2.cfg.state_size == sizeof(double),
                "Pattern 1: state_size mismatch");
    
    /* Cleanup */
    free(checkpoint_blob);
    fastpf_free(&pf1);
    fastpf_free(&pf2);
    
    return 1;
}

/* ========================================================================
 * Test 3: Pattern 2 (load into initialized PF)
 * ======================================================================== */

int test_checkpoint_pattern2_validate_and_overwrite(void)
{
    rw_model_ctx_t model_ctx;
    fastpf_model_t model;
    fastpf_cfg_t cfg;
    fastpf_t pf1, pf2;
    void* checkpoint_blob;
    size_t checkpoint_size;
    int result;
    uint64_t hash_before, hash_after;
    
    /* Setup model */
    model_ctx.process_noise = 0.1;
    model_ctx.meas_noise = 1.0;
    model.ctx = &model_ctx;
    model.prior_sample = rw_prior_sample;
    model.transition_sample = rw_transition_sample;
    model.log_likelihood = rw_log_likelihood;
    model.rejuvenate = NULL;
    
    /* Create and checkpoint PF1 */
    fastpf_cfg_init(&cfg, 50, sizeof(double));
    cfg.rng_seed = SEED_42;
    
    result = fastpf_init(&pf1, &cfg, &model);
    TEST_ASSERT(result == FASTPF_SUCCESS, "PF1 init failed");
    
    checkpoint_size = fastpf_checkpoint_bytes(&pf1);
    checkpoint_blob = malloc(checkpoint_size);
    TEST_ASSERT(checkpoint_blob != NULL, "Allocation failed");
    
    result = fastpf_checkpoint_write(&pf1, checkpoint_blob, checkpoint_size);
    TEST_ASSERT(result == FASTPF_SUCCESS, "Checkpoint write failed");
    
    hash_before = hash_pf_state(&pf1);
    
    /* Create PF2 with different seed (different initial state) */
    cfg.rng_seed = 99999;
    result = fastpf_init(&pf2, &cfg, &model);
    TEST_ASSERT(result == FASTPF_SUCCESS, "PF2 init failed");
    
    /* Pattern 2: Overwrite PF2 state with PF1 checkpoint */
    result = fastpf_checkpoint_read(&pf2, NULL, checkpoint_blob, checkpoint_size);
    TEST_ASSERT(result == FASTPF_SUCCESS,
                "Pattern 2: checkpoint_read failed");
    
    hash_after = hash_pf_state(&pf2);
    
    /* After restore, PF2 state must match PF1 */
    TEST_ASSERT(hash_before == hash_after,
                "Pattern 2: state not correctly overwritten");
    
    /* Cleanup */
    free(checkpoint_blob);
    fastpf_free(&pf1);
    fastpf_free(&pf2);
    
    return 1;
}

/* ========================================================================
 * Test 4: Failure mode - magic bytes mismatch
 * ======================================================================== */

int test_checkpoint_fail_magic_mismatch(void)
{
    rw_model_ctx_t model_ctx;
    fastpf_model_t model;
    fastpf_cfg_t cfg;
    fastpf_t pf1, pf2;
    void* checkpoint_blob;
    size_t checkpoint_size;
    int result;
    
    /* Setup model */
    model_ctx.process_noise = 0.1;
    model_ctx.meas_noise = 1.0;
    model.ctx = &model_ctx;
    model.prior_sample = rw_prior_sample;
    model.transition_sample = rw_transition_sample;
    model.log_likelihood = rw_log_likelihood;
    model.rejuvenate = NULL;
    
    /* Create checkpoint */
    fastpf_cfg_init(&cfg, 50, sizeof(double));
    cfg.rng_seed = SEED_42;
    
    result = fastpf_init(&pf1, &cfg, &model);
    TEST_ASSERT(result == FASTPF_SUCCESS, "PF init failed");
    
    checkpoint_size = fastpf_checkpoint_bytes(&pf1);
    checkpoint_blob = malloc(checkpoint_size);
    TEST_ASSERT(checkpoint_blob != NULL, "Allocation failed");
    
    result = fastpf_checkpoint_write(&pf1, checkpoint_blob, checkpoint_size);
    TEST_ASSERT(result == FASTPF_SUCCESS, "Checkpoint write failed");
    
    /* Corrupt magic bytes */
    ((char*)checkpoint_blob)[0] = 'X';
    
    /* Attempt restore - should fail with magic error */
    memset(&pf2, 0, sizeof(pf2));
    pf2.model = model;
    
    result = fastpf_checkpoint_read(&pf2, &cfg, checkpoint_blob, checkpoint_size);
    TEST_ASSERT(result == FASTPF_ERR_CHECKPOINT_MAGIC,
                "Failed to detect magic mismatch");
    
    /* Cleanup */
    free(checkpoint_blob);
    fastpf_free(&pf1);
    
    return 1;
}

/* ========================================================================
 * Test 5: Failure mode - size mismatch
 * ======================================================================== */

int test_checkpoint_fail_size_mismatch(void)
{
    rw_model_ctx_t model_ctx;
    fastpf_model_t model;
    fastpf_cfg_t cfg1, cfg2;
    fastpf_t pf1, pf2;
    void* checkpoint_blob;
    size_t checkpoint_size;
    int result;
    
    /* Setup model */
    model_ctx.process_noise = 0.1;
    model_ctx.meas_noise = 1.0;
    model.ctx = &model_ctx;
    model.prior_sample = rw_prior_sample;
    model.transition_sample = rw_transition_sample;
    model.log_likelihood = rw_log_likelihood;
    model.rejuvenate = NULL;
    
    /* Create checkpoint with N=50 */
    fastpf_cfg_init(&cfg1, 50, sizeof(double));
    cfg1.rng_seed = SEED_42;
    
    result = fastpf_init(&pf1, &cfg1, &model);
    TEST_ASSERT(result == FASTPF_SUCCESS, "PF1 init failed");
    
    checkpoint_size = fastpf_checkpoint_bytes(&pf1);
    checkpoint_blob = malloc(checkpoint_size);
    TEST_ASSERT(checkpoint_blob != NULL, "Allocation failed");
    
    result = fastpf_checkpoint_write(&pf1, checkpoint_blob, checkpoint_size);
    TEST_ASSERT(result == FASTPF_SUCCESS, "Checkpoint write failed");
    
    /* Try to restore into PF with N=100 (different size) */
    fastpf_cfg_init(&cfg2, 100, sizeof(double));
    cfg2.rng_seed = SEED_42;
    
    result = fastpf_init(&pf2, &cfg2, &model);
    TEST_ASSERT(result == FASTPF_SUCCESS, "PF2 init failed");
    
    result = fastpf_checkpoint_read(&pf2, NULL, checkpoint_blob, checkpoint_size);
    TEST_ASSERT(result == FASTPF_ERR_CHECKPOINT_SIZE,
                "Failed to detect size mismatch");
    
    /* Cleanup */
    free(checkpoint_blob);
    fastpf_free(&pf1);
    fastpf_free(&pf2);
    
    return 1;
}

/* ========================================================================
 * Test 6: Failure mode - truncated buffer
 * ======================================================================== */

int test_checkpoint_fail_truncated_buffer(void)
{
    rw_model_ctx_t model_ctx;
    fastpf_model_t model;
    fastpf_cfg_t cfg;
    fastpf_t pf1, pf2;
    void* checkpoint_blob;
    size_t checkpoint_size;
    int result;
    
    /* Setup model */
    model_ctx.process_noise = 0.1;
    model_ctx.meas_noise = 1.0;
    model.ctx = &model_ctx;
    model.prior_sample = rw_prior_sample;
    model.transition_sample = rw_transition_sample;
    model.log_likelihood = rw_log_likelihood;
    model.rejuvenate = NULL;
    
    /* Create checkpoint */
    fastpf_cfg_init(&cfg, 50, sizeof(double));
    cfg.rng_seed = SEED_42;
    
    result = fastpf_init(&pf1, &cfg, &model);
    TEST_ASSERT(result == FASTPF_SUCCESS, "PF init failed");
    
    checkpoint_size = fastpf_checkpoint_bytes(&pf1);
    checkpoint_blob = malloc(checkpoint_size);
    TEST_ASSERT(checkpoint_blob != NULL, "Allocation failed");
    
    result = fastpf_checkpoint_write(&pf1, checkpoint_blob, checkpoint_size);
    TEST_ASSERT(result == FASTPF_SUCCESS, "Checkpoint write failed");
    
    /* Try to restore with truncated buffer (only half the data) */
    memset(&pf2, 0, sizeof(pf2));
    pf2.model = model;
    
    result = fastpf_checkpoint_read(&pf2, &cfg, checkpoint_blob, checkpoint_size / 2);
    TEST_ASSERT(result == FASTPF_ERR_CHECKPOINT_SIZE,
                "Failed to detect truncated buffer");
    
    /* Cleanup */
    free(checkpoint_blob);
    fastpf_free(&pf1);
    
    return 1;
}

/* ========================================================================
 * Test 7: Failure mode - model callbacks not set
 * ======================================================================== */

int test_checkpoint_fail_no_model_callbacks(void)
{
    rw_model_ctx_t model_ctx;
    fastpf_model_t model;
    fastpf_cfg_t cfg;
    fastpf_t pf1, pf2;
    void* checkpoint_blob;
    size_t checkpoint_size;
    int result;
    
    /* Setup model */
    model_ctx.process_noise = 0.1;
    model_ctx.meas_noise = 1.0;
    model.ctx = &model_ctx;
    model.prior_sample = rw_prior_sample;
    model.transition_sample = rw_transition_sample;
    model.log_likelihood = rw_log_likelihood;
    model.rejuvenate = NULL;
    
    /* Create checkpoint */
    fastpf_cfg_init(&cfg, 50, sizeof(double));
    cfg.rng_seed = SEED_42;
    
    result = fastpf_init(&pf1, &cfg, &model);
    TEST_ASSERT(result == FASTPF_SUCCESS, "PF init failed");
    
    checkpoint_size = fastpf_checkpoint_bytes(&pf1);
    checkpoint_blob = malloc(checkpoint_size);
    TEST_ASSERT(checkpoint_blob != NULL, "Allocation failed");
    
    result = fastpf_checkpoint_write(&pf1, checkpoint_blob, checkpoint_size);
    TEST_ASSERT(result == FASTPF_SUCCESS, "Checkpoint write failed");
    
    /* Try to restore without setting model callbacks (footgun #1) */
    memset(&pf2, 0, sizeof(pf2));
    /* pf2.model NOT set - should fail */
    
    result = fastpf_checkpoint_read(&pf2, &cfg, checkpoint_blob, checkpoint_size);
    TEST_ASSERT(result == FASTPF_ERR_INVALID_ARG,
                "Failed to detect missing model callbacks");
    
    /* Cleanup */
    free(checkpoint_blob);
    fastpf_free(&pf1);
    
    return 1;
}

/* ========================================================================
 * Main test runner
 * ======================================================================== */

int main(void)
{
    int passed = 0;
    int total = 0;
    
    printf("=== Checkpoint Unit Tests ===\n");
    
#define RUN_TEST(test) do { \
    total++; \
    printf("Running " #test "...\n"); \
    if (test()) { \
        printf("PASS: " #test "\n"); \
        passed++; \
    } \
} while(0)
    
    RUN_TEST(test_checkpoint_deterministic_resume);
    RUN_TEST(test_checkpoint_pattern1_load_and_init);
    RUN_TEST(test_checkpoint_pattern2_validate_and_overwrite);
    RUN_TEST(test_checkpoint_fail_magic_mismatch);
    RUN_TEST(test_checkpoint_fail_size_mismatch);
    RUN_TEST(test_checkpoint_fail_truncated_buffer);
    RUN_TEST(test_checkpoint_fail_no_model_callbacks);
    
    printf("\n=== Checkpoint Tests: %d/%d passed ===\n", passed, total);
    
    return (passed == total) ? 0 : 1;
}
