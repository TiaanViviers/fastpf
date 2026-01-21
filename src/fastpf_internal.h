/**
 * @file fastpf_internal.h
 * @brief Internal structures and definitions (not part of public API).
 *
 * This header contains the actual struct definitions that are kept private
 * to allow implementation changes without breaking ABI/API.
 * 
 * NOTE: This header should ONLY be included by:
 *   1. Library source files in src/ directory
 *   2. Test code that needs stack allocation
 *   3. Example code demonstrating advanced usage
 * 
 * Regular users should treat fastpf_t and fastpf_rng_t as truly opaque
 * and use heap allocation or the provided API functions only.
 */

#ifndef FASTPF_INTERNAL_H
#define FASTPF_INTERNAL_H

#include "fastpf.h"

/* ========================================================================
 * Internal RNG structure (opaque to users)
 * ======================================================================== */

struct fastpf_rng_t {
    uint64_t state;  /* PCG32 state */
    uint64_t inc;    /* PCG32 stream selector */
};

/* ========================================================================
 * Internal particle filter structure (opaque to users)
 * ======================================================================== */

struct fastpf_t {
    /* Configuration */
    fastpf_cfg_t cfg;
    fastpf_model_t model;

    /* RNG state */
    fastpf_rng_t rng;                       /**< Master RNG for serial code and resampling. */
    fastpf_rng_t* thread_rngs;              /**< Per-thread RNGs for OpenMP (NULL if not used). */

    /* Particle storage (double buffered) */
    unsigned char* particles_curr;          /**< Current particles: N * state_size bytes. */
    unsigned char* particles_next;          /**< Next particles (scratch): N * state_size bytes. */

    /* Weight storage */
    double* log_weights;                    /**< Log-weights: log w_k^(i). */
    double* norm_weights;                   /**< Normalized weights (always maintained). */

    /* Resampling scratch */
    size_t* resample_indices;               /**< Indices from resampling (scratch). */

    /* Diagnostics */
    fastpf_diagnostics_t diag;

    /* Internal state */
    int initialized;
};

#endif /* FASTPF_INTERNAL_H */
