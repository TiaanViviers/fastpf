/**
 * @file checkpoint.c
 * @brief Particle filter state serialization and restoration.
 *
 * =============================================================================
 * CHECKPOINT SYSTEM DESIGN & PHILOSOPHY
 * =============================================================================
 *
 * Purpose:
 * --------
 * Enable deterministic save/resume of particle filter state for production
 * environments where processes may crash, restart, or transition between
 * training/backtesting and live deployment.
 *
 * Design Goals:
 * -------------
 * 1. DETERMINISM: Restore + continue must be bitwise identical to never stopping
 * 2. SIMPLICITY: Raw binary format, minimal overhead (~80 byte header)
 * 3. SAFETY: Fail loudly on version/platform/size mismatches
 * 4. SEPARATION: PF state only - model params are your layer's responsibility
 *
 * =============================================================================
 * BLOB FORMAT SPECIFICATION (Version 1)
 * =============================================================================
 *
 * Header Layout (80 bytes):
 * -------------------------
 * Offset | Size | Type    | Field              | Purpose
 * -------|------|---------|--------------------|---------------------------------
 *   0    |  8   | char[8] | magic              | "FASTPFCK" (detects corruption)
 *   8    |  4   | uint32  | version            | Format version (currently 1)
 *  12    |  4   | uint32  | endianness_tag     | 0x01020304 (detects byte order)
 *  16    |  4   | uint32  | sizeof_double      | sizeof(double) on save platform
 *  20    |  4   | uint32  | flags              | bit0: has_particles, bit1: has_log_weights
 *  24    |  8   | uint64  | n_particles        | Number of particles N
 *  32    |  8   | uint64  | state_size         | Size of each particle state
 *  40    |  4   | uint32  | resample_method    | FASTPF_RESAMPLE_SYSTEMATIC, etc.
 *  44    |  4   | uint32  | padding            | Reserved (alignment)
 *  48    |  8   | double  | resample_threshold | ESS/N threshold for resampling
 *  56    |  8   | uint64  | rng_state          | PCG32 state
 *  64    |  8   | uint64  | rng_inc            | PCG32 stream selector
 *  72    |  8   | uint64  | reserved           | Future use
 *
 * Payload Layout (variable size):
 * -------------------------------
 * Offset       | Size                  | Field
 * -------------|-----------------------|------------------------------------
 * 80           | N * state_size        | particles_curr (raw binary)
 * 80 + N*ss    | N * sizeof(double)    | log_weights (array of doubles)
 *
 * Total Size: 80 + N * state_size + N * sizeof(double) bytes
 *
 * =============================================================================
 * WHAT IS SAVED
 * =============================================================================
 *
 * SAVED - Core State (required for deterministic continuation):
 * -------------------------------------------------------------
 * cfg.n_particles        - Defines particle array size
 * cfg.state_size         - Defines per-particle memory layout
 * cfg.resample_threshold - Affects future resampling decisions
 * cfg.resample_method    - Affects future resampling behavior
 * rng.state, rng.inc     - Complete RNG state (16 bytes)
 * particles_curr         - Current particle population
 * log_weights            - Current log-weights (unnormalized)
 *
 * NOT SAVED - Scratch/Derived Data (recomputed as needed):
 * ---------------------------------------------------------
 * particles_next         - Scratch buffer, refilled each step
 * norm_weights           - Derived from log_weights via normalize
 * resample_indices       - Scratch buffer, refilled during resample
 * diagnostics (ess, etc) - Recomputed on next step
 * thread_rngs            - Per-thread RNG state (OpenMP only, reinitialized)
 *
 * NOT SAVED - Configuration/Policy (caller's responsibility):
 * -----------------------------------------------------------
 * cfg.rng_seed           - Only relevant at init, RNG state is truth
 * cfg.num_threads        - Runtime policy, not filter state
 * cfg.flags              - Reserved for future use
 *
 * NOT SAVED - Model Callbacks (SV layer's responsibility):
 * --------------------------------------------------------
 * model.ctx              - Model parameters
 * model.prior_sample     - Function pointer
 * model.transition_sample
 * model.log_likelihood
 * model.rejuvenate
 *
 * =============================================================================
 * PORTABILITY & COMPATIBILITY
 * =============================================================================
 *
 * Version 1 Guarantees:
 * ---------------------
 * Same endianness      - Checked via 0x01020304 tag
 * Same sizeof(double)  - Checked (typically 8 bytes)
 * Same architecture    - Implied by above checks
 * Same compiler        - Not checked, but typical in controlled environments
 *
 * Version 1 Limitations:
 * ----------------------
 * No byte-swapping     - Cross-endian load will fail (intentional)
 * No compression       - Raw binary (add in v2 if needed)
 * No CRC/checksum      - Trust filesystem/network layer (add in v2 if needed)
 * No forward compat    - v2 blobs will fail on v1 reader (version check)
 *
 * Future Versioning Strategy:
 * ---------------------------
 * - Version field allows rejecting unknown formats
 * - Flags field (currently bits 0-1 used) allows optional extensions in v2+
 * - Reserved header bytes (8 bytes at offset 72) for future metadata
 * - If v2 adds fields, can use flags to indicate presence and skip gracefully
 *
 * =============================================================================
 * CRITICAL USAGE FOOTGUNS (READ THIS!)
 * =============================================================================
 *
 * FOOTGUN #1: Model state is NOT in the checkpoint
 * ---------------------------------------------------
 * The checkpoint contains ONLY particle filter state. If your model has
 * parameters, YOU must save
 * those separately and reconstruct the model before calling checkpoint_read.
 *
 * FOOTGUN #2: Checkpoint size depends on N and state_size
 * -----------------------------------------------------------
 * You CANNOT load a checkpoint into a PF with different N or state_size.
 * This is checked and will return FASTPF_ERR_CHECKPOINT_SIZE.
 *
 * Example:
 *   // Training with N=1000
 *   fastpf_init(&pf, &cfg_train, &model);  // N=1000
 *   save_checkpoint(&pf, "ckpt.bin");
 *
 *   // Production with N=10000 - THIS WILL FAIL!
 *   fastpf_init(&pf2, &cfg_prod, &model);  // N=10000
 *   load_checkpoint(&pf2, "ckpt.bin");     // ERROR: size mismatch!
 *
 * Solution: Use same N/state_size in training and production, OR retrain
 * with production config before saving checkpoint.
 *
 * FOOTGUN #3: Cross-platform loading is NOT supported (v1)
 * ------------------------------------------------------------
 * Checkpoints are NOT portable across:
 * - Different endianness (x86 vs some ARM)
 * - Different sizeof(double) (rare, but possible on exotic platforms)
 *
 * This is INTENTIONAL for v1 simplicity. The code detects mismatches and
 * fails with FASTPF_ERR_CHECKPOINT_PORTABILITY.
 *
 * If you need cross-platform, wait for v2 (or use text-based serialization).
 *
 * FOOTGUN #4: Checkpoint does NOT include thread configuration
 * ----------------------------------------------------------------
 * cfg.num_threads is a runtime policy, not filter state. After restore,
 * you can run with a different thread count without affecting determinism
 * (the master RNG state is what matters, and thread_rngs are derived).
 *
 * However, if OpenMP is enabled, thread_rngs will be reallocated on restore
 * using the current num_threads setting (or auto-detected if num_threads==0).
 *
 * See tests/unit/test_checkpoint.c for full test suite.
 */

#include "fastpf.h"
#include "fastpf_internal.h"
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

#ifdef FASTPF_USE_OPENMP
    #include <omp.h>
#endif

#define CHECKPOINT_VERSION 1
#define ENDIANNESS_TAG 0x01020304UL
static const char CHECKPOINT_MAGIC[8] = "FASTPFCK";
#define FLAG_HAS_PARTICLES    (1U << 0)
#define FLAG_HAS_LOG_WEIGHTS  (1U << 1)
#ifndef isnan
    #define isnan(x) ((x) != (x))
#endif

/**
 * @brief Checkpoint file header (80 bytes, fixed size).
 *
 * This struct defines the binary layout at the start of every checkpoint blob.
 * It MUST remain stable across versions (v1, v2, ...) - never reorder fields!
 *
 * Padding/alignment: Assumes natural alignment (uint64 on 8-byte boundaries).
 * If you port to exotic platforms, verify with static_assert or offsetof checks.
 */
typedef struct {
    char magic[8];                  /* "FASTPFCK" */
    uint32_t version;               /* Format version (1) */
    uint32_t endianness_tag;        /* 0x01020304 */
    uint32_t sizeof_double;         /* sizeof(double) on save platform */
    uint32_t flags;                 /* bit0: has_particles, bit1: has_log_weights */
    uint64_t n_particles;           /* Number of particles */
    uint64_t state_size;            /* Size of each particle state */
    uint32_t resample_method;       /* FASTPF_RESAMPLE_SYSTEMATIC, etc. */
    uint32_t padding;               /* Reserved for alignment */
    double resample_threshold;      /* ESS/N threshold */
    uint64_t rng_state;             /* PCG32 state */
    uint64_t rng_inc;               /* PCG32 stream selector */
    uint64_t reserved;              /* Future use */
} checkpoint_header_t;

/* ========================================================================
 * Public API Implementation
 * ======================================================================== */

size_t fastpf_checkpoint_bytes(const fastpf_t* pf)
{
    size_t header_size, particles_size, weights_size;
    
    if (pf == NULL || !pf->initialized) {
        return 0;
    }
    
    header_size = sizeof(checkpoint_header_t);
    particles_size = pf->cfg.n_particles * pf->cfg.state_size;
    weights_size = pf->cfg.n_particles * sizeof(double);
    
    return header_size + particles_size + weights_size;
}

int fastpf_checkpoint_write(const fastpf_t* pf, void* dst, size_t dst_bytes)
{
    checkpoint_header_t header;
    unsigned char* write_ptr;
    size_t required_bytes, particles_size, weights_size;
    
    /* Validate inputs */
    if (pf == NULL || !pf->initialized || dst == NULL) {
        return FASTPF_ERR_INVALID_ARG;
    }
    
    /* Check buffer size */
    required_bytes = fastpf_checkpoint_bytes(pf);
    if (dst_bytes < required_bytes) {
        return FASTPF_ERR_CHECKPOINT_SIZE;
    }
    
    /* Build header */
    memset(&header, 0, sizeof(header));
    memcpy(header.magic, CHECKPOINT_MAGIC, 8);
    header.version = CHECKPOINT_VERSION;
    header.endianness_tag = ENDIANNESS_TAG;
    header.sizeof_double = (uint32_t)sizeof(double);
    header.flags = FLAG_HAS_PARTICLES | FLAG_HAS_LOG_WEIGHTS;
    header.n_particles = (uint64_t)pf->cfg.n_particles;
    header.state_size = (uint64_t)pf->cfg.state_size;
    header.resample_method = (uint32_t)pf->cfg.resample_method;
    header.padding = 0;
    header.resample_threshold = pf->cfg.resample_threshold;
    header.rng_state = pf->rng.state;
    header.rng_inc = pf->rng.inc;
    header.reserved = 0;
    
    /* Write header */
    write_ptr = (unsigned char*)dst;
    memcpy(write_ptr, &header, sizeof(header));
    write_ptr += sizeof(header);
    
    /* Write particles */
    particles_size = pf->cfg.n_particles * pf->cfg.state_size;
    memcpy(write_ptr, pf->particles_curr, particles_size);
    write_ptr += particles_size;
    
    /* Write log-weights */
    weights_size = pf->cfg.n_particles * sizeof(double);
    memcpy(write_ptr, pf->log_weights, weights_size);
    write_ptr += weights_size;
    
    return FASTPF_SUCCESS;
}

int fastpf_checkpoint_read(fastpf_t* pf,
                           const fastpf_cfg_t* cfg,
                           const void* src,
                           size_t src_bytes)
{
    checkpoint_header_t header;
    const unsigned char* read_ptr;
    size_t required_bytes, particles_size, weights_size;
    size_t i;
    int need_init;
    
    /* Validate basic inputs */
    if (pf == NULL || src == NULL) {
        return FASTPF_ERR_INVALID_ARG;
    }
    
    /* Check if we need to initialize (Pattern 1) or just restore (Pattern 2) */
    need_init = !pf->initialized;
    
    if (need_init && cfg == NULL) {
        return FASTPF_ERR_INVALID_ARG;  /* Pattern 1 requires cfg */
    }
    
    /* Verify model callbacks are set (critical precondition) */
    if (pf->model.prior_sample == NULL ||
        pf->model.transition_sample == NULL ||
        pf->model.log_likelihood == NULL) {
        return FASTPF_ERR_INVALID_ARG;  /* Model must be set before restore */
    }
    
    /* Check minimum buffer size for header */
    if (src_bytes < sizeof(checkpoint_header_t)) {
        return FASTPF_ERR_CHECKPOINT_SIZE;
    }
    
    /* Read and validate header */
    read_ptr = (const unsigned char*)src;
    memcpy(&header, read_ptr, sizeof(header));
    read_ptr += sizeof(header);
    
    /* Validate magic bytes */
    if (memcmp(header.magic, CHECKPOINT_MAGIC, 8) != 0) {
        return FASTPF_ERR_CHECKPOINT_MAGIC;
    }
    
    /* Validate version */
    if (header.version != CHECKPOINT_VERSION) {
        return FASTPF_ERR_CHECKPOINT_VERSION;
    }
    
    /* Validate endianness */
    if (header.endianness_tag != ENDIANNESS_TAG) {
        return FASTPF_ERR_CHECKPOINT_PORTABILITY;
    }
    
    /* Validate sizeof(double) */
    if (header.sizeof_double != (uint32_t)sizeof(double)) {
        return FASTPF_ERR_CHECKPOINT_PORTABILITY;
    }
    
    /* Calculate required buffer size from header */
    particles_size = (size_t)header.n_particles * (size_t)header.state_size;
    weights_size = (size_t)header.n_particles * sizeof(double);
    required_bytes = sizeof(checkpoint_header_t) + particles_size + weights_size;
    
    if (src_bytes < required_bytes) {
        return FASTPF_ERR_CHECKPOINT_SIZE;
    }
    
    /* Pattern 1: Allocate buffers without calling fastpf_init */
    if (need_init) {
        size_t total_particle_bytes;
        
        /* Build config from blob + user cfg */
        pf->cfg = *cfg;
        pf->cfg.n_particles = (size_t)header.n_particles;
        pf->cfg.state_size = (size_t)header.state_size;
        pf->cfg.resample_threshold = header.resample_threshold;
        pf->cfg.resample_method = (fastpf_resample_method_t)header.resample_method;
        
        /* Initialize RNG (will be overwritten by checkpoint, but need valid state) */
        fastpf_rng_seed(&pf->rng, cfg->rng_seed);
        
        /* Allocate particle buffers */
        total_particle_bytes = pf->cfg.n_particles * pf->cfg.state_size;
        pf->particles_curr = (unsigned char*)malloc(total_particle_bytes);
        pf->particles_next = (unsigned char*)malloc(total_particle_bytes);
        
        if (pf->particles_curr == NULL || pf->particles_next == NULL) {
            free(pf->particles_curr);
            free(pf->particles_next);
            return FASTPF_ERR_ALLOC;
        }
        
        /* Allocate weight buffers */
        pf->log_weights = (double*)malloc(pf->cfg.n_particles * sizeof(double));
        pf->norm_weights = (double*)malloc(pf->cfg.n_particles * sizeof(double));
        
        if (pf->log_weights == NULL || pf->norm_weights == NULL) {
            free(pf->particles_curr);
            free(pf->particles_next);
            free(pf->log_weights);
            free(pf->norm_weights);
            return FASTPF_ERR_ALLOC;
        }
        
        /* Allocate resampling scratch */
        pf->resample_indices = (size_t*)malloc(pf->cfg.n_particles * sizeof(size_t));
        if (pf->resample_indices == NULL) {
            free(pf->particles_curr);
            free(pf->particles_next);
            free(pf->log_weights);
            free(pf->norm_weights);
            return FASTPF_ERR_ALLOC;
        }
        
        /* Initialize per-thread RNGs if OpenMP enabled */
        pf->thread_rngs = NULL;
#ifdef FASTPF_USE_OPENMP
        {
            int num_threads;
            int tid;
            
            if (cfg->num_threads == 0) {
                num_threads = omp_get_max_threads();
            } else if (cfg->num_threads < 0) {
                num_threads = 1;
                omp_set_num_threads(1);
            } else {
                num_threads = cfg->num_threads;
                omp_set_num_threads(cfg->num_threads);
            }
            
            pf->thread_rngs = (fastpf_rng_t*)malloc(num_threads * sizeof(fastpf_rng_t));
            if (pf->thread_rngs == NULL) {
                free(pf->particles_curr);
                free(pf->particles_next);
                free(pf->log_weights);
                free(pf->norm_weights);
                free(pf->resample_indices);
                return FASTPF_ERR_ALLOC;
            }
            
            /* Seed thread RNGs - will be deterministic based on checkpoint RNG state */
            for (tid = 0; tid < num_threads; tid++) {
                fastpf_rng_seed(&pf->thread_rngs[tid], cfg->rng_seed + (uint64_t)tid + 1ULL);
            }
        }
#endif
        
        /* Mark as initialized (particles/weights will be loaded from checkpoint) */
        pf->initialized = 0;  /* Will be set to 1 at end of function */
    }
    /* Pattern 2: Validate existing PF matches blob */
    else {
        if (pf->cfg.n_particles != (size_t)header.n_particles) {
            return FASTPF_ERR_CHECKPOINT_SIZE;
        }
        if (pf->cfg.state_size != (size_t)header.state_size) {
            return FASTPF_ERR_CHECKPOINT_SIZE;
        }
    }
    
    /* Restore RNG state */
    pf->rng.state = header.rng_state;
    pf->rng.inc = header.rng_inc;
    
    /* Restore particles */
    if (header.flags & FLAG_HAS_PARTICLES) {
        memcpy(pf->particles_curr, read_ptr, particles_size);
        read_ptr += particles_size;
    }
    
    /* Restore log-weights and validate (no NaNs) */
    if (header.flags & FLAG_HAS_LOG_WEIGHTS) {
        memcpy(pf->log_weights, read_ptr, weights_size);
        read_ptr += weights_size;
        
        /* Sanity check: detect NaNs in log-weights */
        for (i = 0; i < pf->cfg.n_particles; i++) {
            if (isnan(pf->log_weights[i])) {
                return FASTPF_ERR_CHECKPOINT_CORRUPT;
            }
        }
    }
    
    /* Update config fields that affect runtime behavior */
    pf->cfg.resample_threshold = header.resample_threshold;
    pf->cfg.resample_method = (fastpf_resample_method_t)header.resample_method;
    
    /* Reset diagnostics (will be recomputed on next step) */
    pf->diag.ess = 0.0;
    pf->diag.max_weight = 0.0;
    pf->diag.log_norm_const = 0.0;
    pf->diag.resampled = 0;
    
    /* Mark as initialized (Pattern 1 already did this via fastpf_init) */
    pf->initialized = 1;
    
    return FASTPF_SUCCESS;
}
