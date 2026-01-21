#!/bin/bash
# Compare serial vs OpenMP performance

set -e

RESULTS_DIR="benchmark_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p "$RESULTS_DIR"

echo "Building serial version..."
make clean > /dev/null 2>&1
make benchmarks > /dev/null 2>&1

echo "Running serial benchmarks..."
./bin/bench_particle_scaling > "$RESULTS_DIR/serial_particle_${TIMESTAMP}.csv"
./bin/bench_model_cost > "$RESULTS_DIR/serial_model_${TIMESTAMP}.csv"

echo "Building OpenMP version..."
make clean > /dev/null 2>&1
make benchmarks USE_OPENMP=1 > /dev/null 2>&1

echo "Running OpenMP benchmarks..."
./bin/bench_particle_scaling > "$RESULTS_DIR/openmp_particle_${TIMESTAMP}.csv"
./bin/bench_model_cost > "$RESULTS_DIR/openmp_model_${TIMESTAMP}.csv"

echo ""
echo "Results saved to $RESULTS_DIR/"
echo ""
echo "=== Particle Scaling Comparison ==="
echo ""
echo "Serial:"
cat "$RESULTS_DIR/serial_particle_${TIMESTAMP}.csv"
echo ""
echo "OpenMP ($(nproc) threads):"
cat "$RESULTS_DIR/openmp_particle_${TIMESTAMP}.csv"
echo ""
echo "=== Model Cost Breakdown ==="
echo ""
echo "Serial:"
cat "$RESULTS_DIR/serial_model_${TIMESTAMP}.csv"
echo ""
echo "OpenMP:"
cat "$RESULTS_DIR/openmp_model_${TIMESTAMP}.csv"
