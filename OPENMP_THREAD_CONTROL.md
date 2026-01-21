# OpenMP Thread Control

The fastpf library supports runtime control of OpenMP thread count through two methods.

## Method 1: Environment Variable (Recommended)

Use the standard OpenMP environment variable:

```bash
# Run with 4 threads
export OMP_NUM_THREADS=4
./your_program

# Or inline
OMP_NUM_THREADS=4 ./your_program
```

**Benefits:**
- Standard OpenMP approach
- No code changes needed
- Works with any OpenMP application
- Easy to experiment with different thread counts

## Method 2: API Configuration

Set `cfg.num_threads` before calling `fastpf_init()`:

```c
fastpf_cfg_t cfg;
fastpf_cfg_init(&cfg, N, sizeof(state_t));

cfg.num_threads = 4;  /* Use 4 threads */
/* Or: cfg.num_threads = 0;   Auto (use omp_get_max_threads) */
/* Or: cfg.num_threads = -1;  Force serial even if OpenMP enabled */

fastpf_init(&pf, &cfg, &model);
```

**Values:**
- `num_threads = 0` : Auto-detect (uses `omp_get_max_threads()`) [default]
- `num_threads > 0` : Use explicit thread count
- `num_threads = -1` : Force serial execution

**Note:** `OMP_NUM_THREADS` environment variable takes precedence over `cfg.num_threads = 0`.

## Benchmarking Thread Performance

```bash
# Build with OpenMP
make clean && make benchmarks USE_OPENMP=1

# Test different thread counts
for T in 1 2 4 8 16; do
    echo "=== $T threads ==="
    OMP_NUM_THREADS=$T ./bin/bench_particle_scaling
done
```

## Recommendations

### Real-Time / Low-Latency Use Cases
- **Use serial** (`OMP_NUM_THREADS=1` or `cfg.num_threads = -1`)
- Thread overhead (~200µs) exceeds benefits for single-step updates
- Better cache locality and predictable timing

### Batch Processing / Historical Data
- **Use 4-8 threads** (`OMP_NUM_THREADS=4`)
- Beyond 8 threads, memory bandwidth becomes bottleneck
- Diminishing returns due to Amdahl's Law

### Particle Count Guidelines
- **N < 2000**: Serial is faster (thread overhead dominates)
- **N = 2000-10000**: 2-4 threads optimal
- **N > 10000**: 4-8 threads optimal

## Example

See `examples/example_thread_control.c` for a complete demonstration:

```bash
make examples USE_OPENMP=1
./bin/example_thread_control
```

## Disabling OpenMP Entirely

To build without OpenMP support (serial-only):

```bash
make clean
make lib      # No USE_OPENMP=1 flag
make test
```

The library automatically falls back to serial execution when not compiled with OpenMP.
