# fastpf Benchmarks

Modular, focused performance benchmarks for fastpf library.

## Quick Start

```bash
# Build all benchmarks
make benchmarks

# Run individual benchmarks
./bin/bench_particle_scaling
./bin/bench_observation_scaling
./bin/bench_resampling
./bin/bench_model_cost

# Compare serial vs OpenMP
./scripts/bench_compare.sh

# Profile with gprof
./scripts/profile.sh
```

## Benchmarks

### bench_particle_scaling
Measures performance scaling with particle count (N=100 to 100K).
- **Usage**: `./bin/bench_particle_scaling [T]`
- **Output**: CSV (N, T, total_us, per_step_us, ess, resamples)
- **Purpose**: Verify O(N) complexity, find optimal N for hardware

### bench_observation_scaling  
Measures performance scaling with observation count (T=100 to 100K).
- **Usage**: `./bin/bench_observation_scaling [N]`
- **Output**: CSV (N, T, total_us, per_step_us, ess, resamples)
- **Purpose**: Verify O(T) complexity, test long sequences

### bench_resampling
Measures resampling overhead by varying threshold (0.0 to 1.0).
- **Usage**: `./bin/bench_resampling [N] [T]`
- **Output**: CSV (N, T, threshold, total_us, per_step_us, ess, resamples)
- **Purpose**: Optimize resample_threshold for performance

### bench_model_cost
Isolates library overhead vs user model cost.
- **Usage**: `./bin/bench_model_cost [N] [T]`
- **Output**: CSV (model, N, T, total_us, per_step_us)
- **Purpose**: Identify optimization target (library vs model)

