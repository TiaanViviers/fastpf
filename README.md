# fastpf

A high-performance, model-agnostic Sequential Importance Resampling (SIR) particle filter library implemented in C90-style C with widely supported compiler extensions (64-bit integers). Designed for real-time applications requiring fast, numerically stable state estimation with minimal dependencies.

## Features

- **Model-Agnostic Design**: Generic callback interface supports any state-space model
- **High Performance**: Optimized RNG with Ziggurat algorithm, OpenMP parallelization, adaptive resampling
- **Numerical Stability**: Log-space weight computation with robust logsumexp implementation
- **C90 Compliance**: Portable C (C90-style, requires 64-bit unsigned long long) for maximum portability
- **Zero Dependencies**: Only requires standard C library and math library
- **Thread Control**: Runtime-configurable OpenMP thread count for performance tuning
- **Production Ready**: Comprehensive test suite with unit and accuracy tests

## Performance

Benchmarked on a modern multi-core system:

- **RNG Performance**: 6.48 ns per sample (154M samples/sec) using Ziggurat method
- **Filter Throughput**: 290 μs per step for N=10,000 particles
- **Scaling**: Linear scaling with particle count and observation sequence length

*Performance depends strongly on the cost of the user-provided model callbacks and hardware characteristics.*

## Installation

### Building the Library

```bash
make
```

This produces `bin/libfastpf.a` which can be linked into your application.

### Running Tests

```bash
make test
```

Runs unit tests (RNG, resampling, statistics) and accuracy tests (LGSSM vs Kalman filter).

### Building Examples

```bash
make examples
```

Compiles example programs in `examples/`.

### Running Benchmarks

```bash
make benchmarks
./bin/bench_particle_scaling
./bin/bench_rng
```

## Quick Start

### Basic Usage

```c
#include "fastpf.h"

/* Define your state-space model */
typedef struct {
    double F, Q, H, R;  /* Model parameters */
} my_model_ctx_t;

void my_prior_sample(void* ctx, void* x, fastpf_rng_t* rng) {
    double* state = (double*)x;
    state[0] = fastpf_rng_normal(rng);
}

void my_transition_sample(void* ctx, const void* x_prev, 
                          void* x_next, fastpf_rng_t* rng) {
    my_model_ctx_t* model = (my_model_ctx_t*)ctx;
    const double* x_k_1 = (const double*)x_prev;
    double* x_k = (double*)x_next;
    
    x_k[0] = model->F * x_k_1[0] + sqrt(model->Q) * fastpf_rng_normal(rng);
}

double my_log_likelihood(void* ctx, const void* x, const void* y) {
    my_model_ctx_t* model = (my_model_ctx_t*)ctx;
    const double* state = (const double*)x;
    const double* obs = (const double*)y;
    double residual = obs[0] - model->H * state[0];
    
    return -0.5 * (log(2.0 * 3.14159265359 * model->R) + 
                   (residual * residual) / model->R);
}

int main(void) {
    fastpf_t pf;
    fastpf_cfg_t cfg;
    fastpf_model_t model;
    my_model_ctx_t model_ctx = {0.9, 1.0, 1.0, 1.0};
    double observation;
    int i;
    
    /* Configure particle filter */
    fastpf_cfg_init(&cfg, 1000, sizeof(double));
    cfg.rng_seed = 42;
    cfg.resample_threshold = 0.5;
    cfg.num_threads = 0;  /* Auto-detect */
    
    /* Set up model callbacks */
    model.ctx = &model_ctx;
    model.prior_sample = my_prior_sample;
    model.transition_sample = my_transition_sample;
    model.log_likelihood = my_log_likelihood;
    model.rejuvenate = NULL;  /* Optional */
    
    /* Initialize particle filter */
    if (fastpf_init(&pf, &cfg, &model) != FASTPF_SUCCESS) {
        return 1;
    }
    
    /* Run filter on observation sequence */
    for (i = 0; i < 100; i++) {
        observation = /* ... get observation ... */;
        fastpf_step(&pf, &observation);
        
        /* Access results */
        const fastpf_diagnostics_t* diag = fastpf_get_diagnostics(&pf);
        printf("ESS: %.2f, Resampled: %d\n", diag->ess, diag->resampled);
    }
    
    /* Cleanup */
    fastpf_free(&pf);
    return 0;
}
```

### Compiling Your Application

```bash
gcc -O3 -Iinclude myapp.c -Lbin -lfastpf -lm -o myapp
```

With OpenMP support:

```bash
gcc -O3 -fopenmp -DFASTPF_USE_OPENMP -Iinclude myapp.c -Lbin -lfastpf -lm -o myapp
```

## API Reference

### Configuration

```c
void fastpf_cfg_init(fastpf_cfg_t* cfg, size_t n_particles, size_t state_size);
```

Initialize configuration with sensible defaults. Parameters:
- `n_particles`: Number of particles (typically 100-100,000)
- `state_size`: Size of state vector in bytes

Configuration fields:
- `rng_seed`: Random seed (default: 42)
- `resample_threshold`: ESS/N threshold for adaptive resampling (default: 0.5)
- `resample_method`: Resampling algorithm (default: systematic)
- `num_threads`: OpenMP thread count (0=auto, -1=serial, >0=explicit)

### Particle Filter Operations

```c
int fastpf_init(fastpf_t* pf, const fastpf_cfg_t* cfg, const fastpf_model_t* model);
```

Initialize particle filter with given configuration and model. Returns `FASTPF_SUCCESS` on success.

```c
int fastpf_step(fastpf_t* pf, const void* y_k);
```

Perform one filtering step with observation `y_k`:
1. Propagate particles through transition model
2. Update weights with likelihood
3. Normalize weights and compute ESS
4. Adaptive resampling if ESS < threshold
5. Optional rejuvenation

```c
void fastpf_free(fastpf_t* pf);
```

Free all allocated resources.

### Accessors

```c
const void* fastpf_get_particle(const fastpf_t* pf, size_t i);
const double* fastpf_get_weights(const fastpf_t* pf);
const fastpf_diagnostics_t* fastpf_get_diagnostics(const fastpf_t* pf);
size_t fastpf_num_particles(const fastpf_t* pf);
size_t fastpf_state_size(const fastpf_t* pf);
```

### Model Interface

Users must implement:

```c
typedef struct {
    void* ctx;  /* User context pointer */
    
    /* Sample from prior p(x_0) */
    void (*prior_sample)(void* ctx, void* x_0, fastpf_rng_t* rng);
    
    /* Sample from transition p(x_k | x_{k-1}) */
    void (*transition_sample)(void* ctx, const void* x_prev, 
                              void* x_next, fastpf_rng_t* rng);
    
    /* Compute log p(y_k | x_k) */
    double (*log_likelihood)(void* ctx, const void* x_k, const void* y_k);
    
    /* Optional: rejuvenation after resampling */
    void (*rejuvenate)(void* ctx, void* x, fastpf_rng_t* rng);
} fastpf_model_t;
```

### RNG Functions

```c
void fastpf_rng_seed(fastpf_rng_t* rng, uint64_t seed);
uint32_t fastpf_rng_u32(fastpf_rng_t* rng);
double fastpf_rng_uniform01(fastpf_rng_t* rng);
double fastpf_rng_normal(fastpf_rng_t* rng);
```

The RNG uses PCG32 as the base generator with Ziggurat method for normal sampling.

### Diagnostics

```c
typedef struct {
    double ess;              /* Effective sample size */
    double max_weight;       /* Maximum normalized weight */
    double log_norm_const;   /* Log normalizing constant */
    int resampled;           /* 1 if resampling occurred, 0 otherwise */
} fastpf_diagnostics_t;
```

### Ownership and Memory Management

**Particle Filter Ownership:**
- The particle filter owns all internal memory (particles, weights, scratch buffers)
- Memory is allocated during `fastpf_init()` and freed during `fastpf_free()`
- Users must call `fastpf_free()` to prevent memory leaks

**User Responsibilities:**
- The `fastpf_model_t.ctx` pointer remains owned by the user
- The observation pointer `y_k` passed to `fastpf_step()` must remain valid for the duration of the call
- Observations can be deallocated after `fastpf_step()` returns
- Model callback functions must not free or modify particle memory beyond their designated buffers

**Return Value Lifetimes:**
- `fastpf_get_particle()` returns a pointer valid until the next `fastpf_step()` or `fastpf_free()`
- `fastpf_get_weights()` returns a pointer valid until the next `fastpf_step()` or `fastpf_free()`
- `fastpf_get_diagnostics()` returns a pointer valid until `fastpf_free()`
- Users should copy data if persistence beyond these lifetimes is required

**Weight Semantics:**
- `fastpf_get_weights()` returns the **current normalized weights** at the time of the call
- After resampling: weights are uniform (1/N for all particles)
- Before resampling: weights reflect the observation likelihood and history
- Check `diag->resampled` to determine if weights were reset in the last step
- For custom estimators, query weights immediately after `fastpf_step()` but before the next step

**Thread Safety:**
- Each `fastpf_t` instance is NOT thread-safe; do not call methods concurrently on the same instance
- Multiple `fastpf_t` instances can be used independently in different threads
- Internal OpenMP parallelization is managed automatically within each `fastpf_step()` call


## Thread Control

The library supports runtime control of OpenMP thread count. See [OPENMP_THREAD_CONTROL.md](OPENMP_THREAD_CONTROL.md) for details.

**Methods:**
1. Environment variable: `export OMP_NUM_THREADS=4`
2. Configuration field: `cfg.num_threads = 4`

**Modes:**
- `num_threads = 0`: Auto-detect (uses `omp_get_max_threads()`)
- `num_threads > 0`: Explicit thread count
- `num_threads = -1`: Force serial execution

## State Management & Checkpointing

The library provides a complete checkpoint system for saving and restoring particle filter state. This is essential for production environments where processes may crash, restart, or transition between training/backtesting and live deployment.

### Overview

**Checkpointing enables:**
- **Crash Recovery**: Resume filtering after unexpected termination
- **Model Deployment**: Transfer trained filter state from backtesting to production
- **State Persistence**: Save filter state between runs (e.g., daily batch jobs)

**Determinism Guarantee**: Restoring from a checkpoint and continuing with the same observation sequence produces **bitwise-identical** results to a run that never checkpointed.

### API Functions

```c
/* Calculate required buffer size */
size_t fastpf_checkpoint_bytes(const fastpf_t* pf);

/* Save filter state to binary blob */
int fastpf_checkpoint_write(const fastpf_t* pf, void* dst, size_t dst_bytes);

/* Restore filter state from binary blob */
int fastpf_checkpoint_read(fastpf_t* pf, const fastpf_cfg_t* cfg,
                           const void* src, size_t src_bytes);
```

### What Is Saved

The checkpoint blob contains **only** the particle filter state needed for deterministic continuation:

**Saved (Core State):**
- Configuration: `n_particles`, `state_size`, `resample_threshold`, `resample_method`
- RNG state: Complete PCG32 state (16 bytes) for deterministic continuation
- Particles: All N particle states (`particles_curr` buffer)
- Weights: All N log-weights (unnormalized)

**Not Saved (Recomputed/Scratch):**
- Scratch buffers: `particles_next`, `norm_weights`, `resample_indices`
- Diagnostics: `ess`, `max_weight`, `log_norm_const` (recomputed on next step)
- Thread configuration: `num_threads` (runtime policy, not state)
- Original seed: `rng_seed` (irrelevant after init, RNG state is truth)

**Not Saved (Your Responsibility):**
- **Model callbacks**: Function pointers (must be reconstructed before restore)
- **Model parameters**: `model.ctx` data

### Usage Patterns

#### Pattern 1: Save and Restore (Typical Workflow)

```c
/* During live operation - save checkpoint */
{
    size_t ckpt_size = fastpf_checkpoint_bytes(&pf);
    void* ckpt_blob = malloc(ckpt_size);
    
    int result = fastpf_checkpoint_write(&pf, ckpt_blob, ckpt_size);
    assert(result == FASTPF_SUCCESS);
    
    /* Write to disk/database */
    FILE* f = fopen("pf_state.ckpt", "wb");
    fwrite(ckpt_blob, 1, ckpt_size, f);
    fclose(f);
    
    free(ckpt_blob);
}

/* After restart - restore checkpoint */
{
    /* Step 1: Reconstruct model (CRITICAL!) */
    my_model_ctx_t model_params;
    load_model_params(&model_params, "model.params");  /* Your function */
    
    fastpf_model_t model;
    model.ctx = &model_params;
    model.prior_sample = my_prior_sample;
    model.transition_sample = my_transition_sample;
    model.log_likelihood = my_log_likelihood;
    model.rejuvenate = NULL;
    
    /* Step 2: Load checkpoint blob */
    FILE* f = fopen("pf_state.ckpt", "rb");
    fseek(f, 0, SEEK_END);
    size_t ckpt_size = ftell(f);
    fseek(f, 0, SEEK_SET);
    
    void* ckpt_blob = malloc(ckpt_size);
    fread(ckpt_blob, 1, ckpt_size, f);
    fclose(f);
    
    /* Step 3: Restore PF state */
    fastpf_t pf;
    fastpf_cfg_t cfg;
    
    memset(&pf, 0, sizeof(pf));
    pf.model = model;  /* Set callbacks BEFORE restore */
    
    fastpf_cfg_init(&cfg, 1000, sizeof(double));  /* N, state_size */
    cfg.num_threads = 4;  /* Can differ from original */
    
    int result = fastpf_checkpoint_read(&pf, &cfg, ckpt_blob, ckpt_size);
    assert(result == FASTPF_SUCCESS);
    
    free(ckpt_blob);
    
    /* Step 4: Continue filtering */
    fastpf_step(&pf, &next_observation);
    
    /* Cleanup */
    fastpf_free(&pf);
}
```

#### Pattern 2: Overwrite Existing PF

```c
/* Restore into already-initialized PF (overwrites state) */
fastpf_t pf;
fastpf_cfg_t cfg;
fastpf_model_t model;

/* Initialize PF normally */
fastpf_cfg_init(&cfg, 1000, sizeof(double));
/* ... set up model ... */
fastpf_init(&pf, &cfg, &model);

/* Load checkpoint (validates config matches, then overwrites state) */
/* Note: cfg can be NULL in Pattern 2 since pf is already initialized */
int result = fastpf_checkpoint_read(&pf, NULL, ckpt_blob, ckpt_size);
assert(result == FASTPF_SUCCESS);

/* PF now contains checkpoint state, ready to continue */
fastpf_step(&pf, &observation);
```

### Critical Preconditions

**FOOTGUN #1: Model callbacks MUST be set before `fastpf_checkpoint_read()`**

The checkpoint does **not** contain model callbacks or `model.ctx` data. You **must**:
1. Separately save/load your model parameters
2. Reconstruct the `fastpf_model_t` structure with valid function pointers
3. Set `pf.model` **before** calling `fastpf_checkpoint_read()`

Failure to do this returns `FASTPF_ERR_INVALID_ARG`.

**FOOTGUN #2: Checkpoint size depends on N and state_size**

You **cannot** load a checkpoint saved with N=1000 into a PF with N=10000. The code validates this and returns `FASTPF_ERR_CHECKPOINT_SIZE` on mismatch.

**Solution**: Use the same `n_particles` and `state_size` in training and production.

**FOOTGUN #3: Cross-platform checkpoints not supported (v1)**

Checkpoints are **not portable** across:
- Different endianness (x86 vs some ARM variants)
- Different `sizeof(double)` (rare, but possible on exotic platforms)

The code detects mismatches and returns `FASTPF_ERR_CHECKPOINT_PORTABILITY`. This is **intentional** for v1 simplicity - no byte-swapping overhead.

### Blob Format (Version 1)

The checkpoint blob has the following structure:

**Header (80 bytes):**
```
Offset | Size | Field              | Purpose
-------|------|--------------------|---------------------------------
0      | 8    | magic              | "FASTPFCK" (corruption detection)
8      | 4    | version            | Format version (currently 1)
12     | 4    | endianness_tag     | 0x01020304 (byte order check)
16     | 4    | sizeof_double      | Platform verification
20     | 4    | flags              | Optional payload indicators
24     | 8    | n_particles        | Number of particles N
32     | 8    | state_size         | Particle state size
40     | 4    | resample_method    | Resampling algorithm
48     | 8    | resample_threshold | ESS/N threshold
56     | 8    | rng_state          | PCG32 state
64     | 8    | rng_inc            | PCG32 stream selector
72     | 8    | reserved           | Future use
```

**Payload (variable size):**
```
80                : particles_curr[N * state_size]
80 + N*state_size : log_weights[N]
```

**Total size**: `80 + N * state_size + N * sizeof(double)` bytes

### Error Codes

```c
FASTPF_SUCCESS                      /* 0:  Checkpoint saved/loaded */
FASTPF_ERR_INVALID_ARG              /* -1: NULL pointer or model not set */
FASTPF_ERR_ALLOC                    /* -2: Memory allocation failed */
FASTPF_ERR_CHECKPOINT_MAGIC         /* -10: Magic bytes mismatch */
FASTPF_ERR_CHECKPOINT_VERSION       /* -11: Unsupported version */
FASTPF_ERR_CHECKPOINT_PORTABILITY   /* -12: Endianness/sizeof mismatch */
FASTPF_ERR_CHECKPOINT_SIZE          /* -13: Size mismatch or truncated */
FASTPF_ERR_CHECKPOINT_CORRUPT       /* -14: NaNs or invalid data */
```

### Testing & Validation

The checkpoint system includes comprehensive tests (see `tests/unit/test_checkpoint.c`):

1. **Deterministic resume**: Hash-based verification that checkpoint→resume produces bitwise-identical results
2. **Resampling alignment**: Both paths resample on the same steps after restore
3. **Pattern 1 & 2**: Both usage patterns tested
4. **Failure modes**: All validation checks tested (magic, version, size, endianness, model callbacks)

Run checkpoint tests:
```bash
make test  # Includes bin/test_checkpoint
```

### Future Extensions (v2+)

The checkpoint format is designed for extensibility:
- `version` field allows rejecting unknown formats
- `flags` field (bits 0-1 used) supports optional extensions
- `reserved` bytes (8 bytes) for future metadata
- Could add: compression, CRC checksums, byte-swapping for portability

## Algorithm Details

### Sequential Importance Resampling (SIR)

The library implements the bootstrap particle filter with the following steps:

1. **Initialization**: Sample N particles from prior distribution
2. **Prediction**: Propagate particles through transition model
3. **Update**: Weight particles by observation likelihood
4. **Normalization**: Convert log-weights to normalized weights
5. **Resampling**: Systematic resampling when ESS < threshold × N
6. **Rejuvenation**: Optional rejuvenation hook (e.g. MCMC move) to increase particle diversity

### Adaptive Resampling

Resampling only occurs when the effective sample size (ESS) drops below a threshold:

```
ESS = 1 / Σ(w_i²)
```

Default threshold is 0.5N. This avoids unnecessary resampling and particle depletion.

### Numerical Stability

- **Log-space weights**: Avoids underflow in likelihood computation
- **Logsumexp trick**: Numerically stable normalization
- **Robust ESS**: Handles edge cases (zero weights, numerical precision)
- **Infinity handling**: Graceful handling of -∞ log-likelihoods

### RNG Implementation

- **Base generator**: PCG32 (Permuted Congruential Generator)
- **Normal sampling**: Ziggurat algorithm (Marsaglia & Tsang, 2000)
- **Performance**: 3× faster than Box-Muller transform
- **Quality**: Passes statistical tests, suitable for Monte Carlo applications

## Project Structure

```
.
├── benchmarks/          # Performance benchmarks
│   ├── bench_*.c        # Individual benchmark programs
│   └── bench_common.*   # Shared benchmark utilities
├── examples/            # Example programs
│   ├── example_random_walk.c
│   └── example_thread_control.c
├── include/             # Public API
│   └── fastpf.h         # Main header file
├── scripts/             # Utility scripts
│   ├── bench_compare.sh # Benchmark comparison tool
│   └── profile.sh       # Profiling with gprof
├── src/                 # Implementation
│   ├── fastpf.c         # Core particle filter
│   ├── fastpf_internal.h
│   ├── resample.c       # Resampling algorithms
│   ├── rng.c            # Random number generation
│   ├── stats.c          # ESS and statistics
│   └── util.c           # Utility functions
└── tests/               # Test suite
    ├── accuracy/        # Accuracy tests vs. ground truth
    └── unit/            # Unit tests for components
```

## Testing

### Unit Tests

- **RNG**: Determinism, statistical moments, distribution properties
- **Resampling**: Index validity, distribution correctness, edge cases
- **Statistics**: ESS computation, weight normalization
- **Utilities**: Logsumexp, numerical stability

### Accuracy Tests

- **LGSSM**: Particle filter vs. Kalman filter comparison
- **Metrics**: RMSE between PF and KF estimates over time

All tests use deterministic seeds for reproducibility.

## Benchmarks

The `benchmarks/` directory contains performance tests:

- `bench_particle_scaling`: Throughput vs. particle count
- `bench_observation_scaling`: Throughput vs. time series length
- `bench_model_cost`: Impact of model complexity
- `bench_resampling`: Resampling algorithm performance
- `bench_rng`: Pure RNG throughput

Run with `make benchmarks` and execute individual benchmarks from `bin/`.

## Examples

### Random Walk Model

See `examples/example_random_walk.c` for a complete working example of a 1D random walk with Gaussian observations.

### Thread Control

See `examples/example_thread_control.c` for demonstration of runtime thread count control and performance comparison.

## Compiler Support

Tested with:
- GCC 4.8+
- Clang 3.5+

## Dependencies

- Standard C library (stdlib, string, math)
- OpenMP (optional, for parallelization)

No external dependencies required.

## Performance Tuning

### Particle Count

- **N < 100**: High variance, poor approximation
- **N = 100-1000**: Good for simple models
- **N = 1000-10000**: Suitable for most applications
- **N > 10000**: Expensive, use for complex/high-dimensional models

### Thread Count

Profile your application to find optimal thread count. Speedup saturates due to synchronization overhead. For small particle counts (N < 500), serial execution may be faster.

### Resampling Threshold

- **threshold = 0.3**: More frequent resampling, less degeneracy
- **threshold = 0.5**: Balanced (default)
- **threshold = 0.7**: Less frequent resampling, more computation per step


## License

See [LICENSE](LICENSE) file for details.


## References

1. Simo Särkkä (2013). *Bayesian Filtering and Smoothing*. Cambridge University Press.

2. Doucet, A., & Johansen, A. M. (2009). A tutorial on particle filtering and smoothing. *Handbook of Nonlinear Filtering*, 12, 656-704.

3. Marsaglia, G., & Tsang, W. W. (2000). The ziggurat method for generating random variables. *Journal of Statistical Software*, 5(8), 1-7.

4. O'Neill, M. E. (2014). PCG: A family of simple fast space-efficient statistically good algorithms for random number generation. *Technical Report*.

5. Gordon, N. J., Salmond, D. J., & Smith, A. F. (1993). Novel approach to nonlinear/non-Gaussian Bayesian state estimation. *IEE Proceedings F*, 140(2), 107-113.
