/**
 * @file fastpf.h
 * @brief High-performance model-agnostic Sequential Importance Resampling (SIR)
 *        Particle Filter library in strict C90.
 *
 * Public API for the fastpf library.
 */

#ifndef FASTPF_H
#define FASTPF_H

#include <stddef.h> /* size_t */

/* C90-compatible fixed-width integer types */
#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 199901L
    #include <stdint.h>
#else
    /* Fallback: assumes compiler supports 64-bit unsigned long long.
     * This is a C99 feature but widely supported as an extension in C90 compilers.
     * If your compiler lacks this, you'll need to provide uint64_t another way. */
    typedef unsigned int       uint32_t;
    typedef unsigned long long uint64_t;
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================
 * Error codes
 * ======================================================================== */

#define FASTPF_SUCCESS           0
#define FASTPF_ERR_INVALID_ARG  -1
#define FASTPF_ERR_ALLOC        -2
#define FASTPF_ERR_NUMERICAL    -3
#define FASTPF_ERR_NOT_INIT     -4

/* Checkpointing error codes */
#define FASTPF_ERR_CHECKPOINT_MAGIC        -10  /* Magic bytes mismatch */
#define FASTPF_ERR_CHECKPOINT_VERSION      -11  /* Unsupported version */
#define FASTPF_ERR_CHECKPOINT_PORTABILITY  -12  /* Endianness/sizeof mismatch */
#define FASTPF_ERR_CHECKPOINT_SIZE         -13  /* Size mismatch or truncated */
#define FASTPF_ERR_CHECKPOINT_CORRUPT      -14  /* NaNs or invalid data */

/* ========================================================================
 * Forward declarations
 * ======================================================================== */

typedef struct fastpf_rng_t fastpf_rng_t;
typedef struct fastpf_t fastpf_t;

/* ========================================================================
 * RNG interface
 * ======================================================================== */

/**
 * @brief Opaque RNG state (PCG32 implementation).
 * Definition is private; use the provided API functions only.
 */
/* struct fastpf_rng_t: definition in src/fastpf_internal.h */

/**
 * @brief Seed the RNG with a 64-bit seed.
 * @param rng Pointer to RNG state.
 * @param seed 64-bit seed value.
 */
void fastpf_rng_seed(fastpf_rng_t* rng, uint64_t seed);

/**
 * @brief Generate a uniformly distributed 32-bit unsigned integer.
 * @param rng Pointer to RNG state.
 * @return Random 32-bit unsigned integer.
 */
unsigned int fastpf_rng_u32(fastpf_rng_t* rng);

/**
 * @brief Generate a uniformly distributed double in (0, 1).
 * @param rng Pointer to RNG state.
 * @return Random double in the open interval (0, 1).
 */
double fastpf_rng_uniform01(fastpf_rng_t* rng);

/**
 * @brief Generate a standard normal (Gaussian) variate N(0,1).
 * @param rng Pointer to RNG state.
 * @return Random standard normal variate.
 */
double fastpf_rng_normal(fastpf_rng_t* rng);

/* ========================================================================
 * Model callback interface
 * ======================================================================== */

/**
 * @brief Model callback: sample from the prior p(x_0).
 * @param ctx User-provided model context.
 * @param x0_out Output buffer (size state_size bytes) to write sampled x_0.
 * @param rng RNG state for sampling.
 */
typedef void (*fastpf_prior_sample_fn)(void* ctx, void* x0_out, fastpf_rng_t* rng);

/**
 * @brief Model callback: sample from transition p(x_k | x_{k-1}).
 * @param ctx User-provided model context.
 * @param x_prev Previous state (size state_size bytes).
 * @param x_out Output buffer (size state_size bytes) for new state.
 * @param rng RNG state for sampling.
 */
typedef void (*fastpf_transition_sample_fn)(void* ctx, const void* x_prev, void* x_out, fastpf_rng_t* rng);

/**
 * @brief Model callback: compute log-likelihood log p(y_k | x_k).
 * @param ctx User-provided model context.
 * @param x Current state (size state_size bytes).
 * @param y Observation at time k (user-defined structure).
 * @return Log-likelihood value.
 */
typedef double (*fastpf_log_likelihood_fn)(void* ctx, const void* x, const void* y);

/**
 * @brief Optional callback: rejuvenation/roughening after resampling.
 * @param ctx User-provided model context.
 * @param x Particle state to perturb in-place.
 * @param rng RNG state.
 */
typedef void (*fastpf_rejuvenate_fn)(void* ctx, void* x, fastpf_rng_t* rng);

/**
 * @brief Model specification: function pointers + context.
 */
typedef struct {
    void* ctx;                              /**< User-provided context pointer. */
    fastpf_prior_sample_fn prior_sample;
    fastpf_transition_sample_fn transition_sample;
    fastpf_log_likelihood_fn log_likelihood;
    fastpf_rejuvenate_fn rejuvenate;        /**< Optional; may be NULL. */
} fastpf_model_t;

/* ========================================================================
 * Resampling methods
 * ======================================================================== */

typedef enum {
    FASTPF_RESAMPLE_SYSTEMATIC = 0,
    FASTPF_RESAMPLE_STRATIFIED = 1
} fastpf_resample_method_t;

/* ========================================================================
 * Configuration
 * ======================================================================== */

/**
 * @brief Particle filter configuration.
 */
typedef struct {
    size_t n_particles;                     /**< Number of particles N. */
    size_t state_size;                      /**< Size of state in bytes. */
    uint64_t rng_seed;                      /**< RNG seed for reproducibility. */
    double resample_threshold;              /**< Resample when ESS/N < threshold. Valid range: (0, 1], default 0.5. */
    fastpf_resample_method_t resample_method;
    int num_threads;                        /**< OpenMP thread count. 0=auto (omp_get_max_threads), -1=serial, >0=explicit count */
    unsigned int flags;                     /**< Future feature flags (reserved, set to 0). */
} fastpf_cfg_t;

/**
 * @brief Initialize configuration with sensible defaults.
 * @param cfg Configuration structure to initialize.
 * @param n_particles Number of particles.
 * @param state_size Size of state in bytes.
 */
void fastpf_cfg_init(fastpf_cfg_t* cfg, size_t n_particles, size_t state_size);

/* ========================================================================
 * Diagnostics (per-step)
 * ======================================================================== */

/**
 * @brief Per-step diagnostics structure.
 */
typedef struct {
    double ess;                             /**< Effective sample size. */
    double max_weight;                      /**< Maximum normalized weight. */
    double log_norm_const;                  /**< Log of the normalizing constant (sum of unnormalized weights). */
    int resampled;                          /**< 1 if resampling occurred this step, 0 otherwise. */
} fastpf_diagnostics_t;

/* ========================================================================
 * Particle Filter instance
 * ======================================================================== */

/**
 * @brief Opaque particle filter instance.
 * Definition is private; use the provided API functions and accessors only.
 */
/* struct fastpf_t: definition in src/fastpf_internal.h */

/* ========================================================================
 * Particle Filter API
 * ======================================================================== */

/**
 * @brief Initialize a particle filter instance.
 * @param pf Pointer to PF instance to initialize.
 * @param cfg Configuration.
 * @param model Model callbacks.
 * @return FASTPF_SUCCESS or error code.
 */
int fastpf_init(fastpf_t* pf, const fastpf_cfg_t* cfg, const fastpf_model_t* model);

/**
 * @brief Free resources held by particle filter.
 * @param pf Pointer to PF instance.
 */
void fastpf_free(fastpf_t* pf);

/**
 * @brief Perform one filtering step: propagate, weight, normalize, ESS, resample.
 * @param pf Pointer to PF instance.
 * @param y_k Observation at time k (user-defined structure pointer).
 * @return FASTPF_SUCCESS or error code.
 */
int fastpf_step(fastpf_t* pf, const void* y_k);

/**
 * @brief Get pointer to current particle i.
 * @param pf Pointer to PF instance.
 * @param i Particle index (0 <= i < N).
 * @return Pointer to particle state (state_size bytes), or NULL if invalid.
 */
const void* fastpf_get_particle(const fastpf_t* pf, size_t i);

/**
 * @brief Get pointer to normalized weights array.
 * @param pf Pointer to PF instance.
 * @return Pointer to array of N normalized weights.
 */
const double* fastpf_get_weights(const fastpf_t* pf);

/**
 * @brief Get diagnostics from the last step.
 * @param pf Pointer to PF instance.
 * @return Pointer to diagnostics structure.
 */
const fastpf_diagnostics_t* fastpf_get_diagnostics(const fastpf_t* pf);

/**
 * @brief Get the number of particles in the filter.
 * @param pf Pointer to PF instance.
 * @return Number of particles N.
 */
size_t fastpf_num_particles(const fastpf_t* pf);

/**
 * @brief Get the state size in bytes.
 * @param pf Pointer to PF instance.
 * @return Size of each particle state in bytes.
 */
size_t fastpf_state_size(const fastpf_t* pf);

/* ========================================================================
 * State Management & Checkpointing
 * ======================================================================== */

/**
 * @brief Calculate the size (in bytes) required for a checkpoint blob.
 *
 * Returns the total number of bytes needed to serialize the complete
 * particle filter state, including:
 * - Header with metadata (magic, version, cfg, RNG state)
 * - All N particles (N * state_size bytes)
 * - All N log-weights (N * sizeof(double) bytes)
 *
 * Use this to allocate a buffer before calling fastpf_checkpoint_write().
 *
 * @param pf Pointer to initialized PF instance.
 * @return Number of bytes required, or 0 if pf is invalid/uninitialized.
 */
size_t fastpf_checkpoint_bytes(const fastpf_t* pf);

/**
 * @brief Serialize particle filter state to a binary blob.
 *
 * Writes a portable-enough binary checkpoint that captures the complete
 * PF state needed for deterministic resume. The blob includes:
 * - Configuration (N, state_size, resample config)
 * - RNG state (for deterministic continuation)
 * - Current particles (particles_curr buffer)
 * - Current log-weights
 *
 * The blob does NOT include:
 * - Scratch buffers (particles_next, norm_weights, resample_indices)
 * - Diagnostics (recomputed on next step)
 * - Model callbacks or model.ctx data
 * - Thread configuration (runtime decision, not state)
 *
 * Portability guarantees:
 * - Same architecture (endianness checked on load)
 * - Same compiler/platform (sizeof(double) checked on load)
 * - Version 1 format: no byte-swapping, raw binary dump
 * - Fails loudly on mismatch rather than silently corrupting
 *
 * CRITICAL: The blob is a snapshot of PF state only.
 *
 * @param pf Pointer to initialized PF instance.
 * @param dst Destination buffer (must be at least fastpf_checkpoint_bytes(pf) bytes).
 * @param dst_bytes Size of destination buffer (for bounds checking).
 * @return FASTPF_SUCCESS or error code:
 *         - FASTPF_ERR_INVALID_ARG if pf is NULL or uninitialized
 *         - FASTPF_ERR_CHECKPOINT_SIZE if dst_bytes is too small
 */
int fastpf_checkpoint_write(const fastpf_t* pf, void* dst, size_t dst_bytes);

/**
 * @brief Restore particle filter state from a binary checkpoint blob.
 *
 * Loads a previously saved checkpoint into a PF instance. Two usage patterns:
 *
 * Pattern 1: Load into uninitialized PF (allocates + restores)
 * -----------------------------------------------------------
 *   fastpf_t pf;
 *   memset(&pf, 0, sizeof(pf));
 *   pf.model = ...; // Set callbacks first!
 *   fastpf_checkpoint_read(&pf, &cfg, blob, blob_size);
 *   // pf is now initialized and ready for fastpf_step()
 *
 * Pattern 2: Load into already-initialized PF (validates + overwrites)
 * ---------------------------------------------------------------------
 *   fastpf_t pf;
 *   fastpf_init(&pf, &cfg, &model);
 *   fastpf_checkpoint_read(&pf, NULL, blob, blob_size);
 *   // pf state overwritten, ready to continue from checkpoint
 *
 * CRITICAL PRECONDITIONS:
 * 1. Model callbacks MUST be set before calling this function.
 *    The checkpoint does NOT contain callbacks.
 *
 * 2. The blob must match the PF configuration:
 *    - Same n_particles
 *    - Same state_size
 *    - Same endianness and sizeof(double)
 *
 * Validation performed:
 * - Magic bytes ("FASTPFCK")
 * - Version (currently only v1 supported)
 * - Endianness tag (0x01020304 in native byte order)
 * - sizeof(double) matches current platform
 * - n_particles and state_size match (if pf already initialized)
 * - Buffer size sufficient
 * - No NaNs in log_weights
 * - Model callbacks are non-NULL
 *
 * After successful restore:
 * - pf->initialized = 1
 * - Particles and log-weights restored
 * - RNG state restored (deterministic continuation)
 * - Diagnostics reset to invalid/zero (recomputed on next step)
 * - Scratch buffers left uninitialized (filled on next step)
 *
 * DETERMINISM GUARANTEE: If you restore from a checkpoint and continue
 * with the same observation sequence, the trajectory will be bitwise
 * identical to a run that never checkpointed.
 *
 * @param pf Pointer to PF instance (may be uninitialized or already init).
 * @param cfg Configuration (required if pf->initialized==0, may be NULL otherwise).
 * @param src Source buffer containing checkpoint blob.
 * @param src_bytes Size of source buffer.
 * @return FASTPF_SUCCESS or error code:
 *         - FASTPF_ERR_INVALID_ARG if pf is NULL, model callbacks not set, or cfg required but NULL
 *         - FASTPF_ERR_CHECKPOINT_MAGIC if magic bytes don't match
 *         - FASTPF_ERR_CHECKPOINT_VERSION if version unsupported
 *         - FASTPF_ERR_CHECKPOINT_PORTABILITY if endianness/sizeof mismatch
 *         - FASTPF_ERR_CHECKPOINT_SIZE if size mismatch or truncated
 *         - FASTPF_ERR_CHECKPOINT_CORRUPT if NaNs or invalid data detected
 *         - FASTPF_ERR_ALLOC if memory allocation fails (Pattern 1 only)
 */
int fastpf_checkpoint_read(fastpf_t* pf,
                           const fastpf_cfg_t* cfg,
                           const void* src,
                           size_t src_bytes);

#ifdef __cplusplus
}
#endif

#endif /* FASTPF_H */
