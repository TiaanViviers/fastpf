#!/bin/bash
# Profile fastpf to identify performance bottlenecks
# Usage: ./scripts/profile.sh [benchmark_name] [profiler]
#   benchmark_name: Name of benchmark to profile (default: benchmark_pf)
#   profiler: perf, gprof, or callgrind (default: auto-detect)

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
BENCHMARK="${1:-benchmark_pf}"
PROFILER="${2:-auto}"
BUILD_DIR="."
PROFILE_DIR="profile_results"

# Create profile results directory
mkdir -p "$PROFILE_DIR"

echo -e "${BLUE}=== fastpf Profiler ===${NC}"
echo "Benchmark: $BENCHMARK"
echo "Profiler: $PROFILER"
echo ""

# Detect available profilers
detect_profilers() {
    HAVE_PERF=0
    HAVE_GPROF=0
    HAVE_VALGRIND=0
    
    if command -v perf &> /dev/null; then
        HAVE_PERF=1
    fi
    
    if command -v gprof &> /dev/null; then
        HAVE_GPROF=1
    fi
    
    if command -v valgrind &> /dev/null; then
        HAVE_VALGRIND=1
    fi
    
    if [ "$PROFILER" = "auto" ]; then
        if [ $HAVE_GPROF -eq 1 ]; then
            PROFILER="gprof"
        elif [ $HAVE_PERF -eq 1 ]; then
            PROFILER="perf"
        elif [ $HAVE_VALGRIND -eq 1 ]; then
            PROFILER="callgrind"
        else
            echo -e "${RED}Error: No profiler found!${NC}"
            echo "Please install one of: perf, gprof, valgrind"
            exit 1
        fi
    fi
}

# Build with profiling support
build_for_profiling() {
    echo -e "${YELLOW}Building with profiling support...${NC}"
    
    case $PROFILER in
        perf)
            # Build with -g for debug symbols, keep -O3 for realistic performance
            echo "Building with: -O3 -g (optimized with debug symbols)"
            make clean > /dev/null 2>&1 || true
            # Override CFLAGS to add -g for debug symbols
            make CFLAGS="-std=c90 -pedantic -Wall -Wextra -Wno-long-long -O3 -g" benchmarks > /dev/null 2>&1
            ;;
        gprof)
            # Build with -pg for gprof profiling
            echo "Building with: -O3 -g -pg (gprof profiling)"
            make clean > /dev/null 2>&1 || true
            # Override CFLAGS to add -g -pg for gprof
            make CFLAGS="-std=c90 -pedantic -Wall -Wextra -Wno-long-long -O3 -g -pg" benchmarks > /dev/null 2>&1
            ;;
        callgrind)
            # Build with -g, no special flags needed
            echo "Building with: -O3 -g (callgrind profiling)"
            make clean > /dev/null 2>&1 || true
            # Override CFLAGS to add -g
            make CFLAGS="-std=c90 -pedantic -Wall -Wextra -Wno-long-long -O3 -g" benchmarks > /dev/null 2>&1
            ;;
    esac
    
    if [ ! -f "bin/$BENCHMARK" ]; then
        echo -e "${RED}Error: Failed to build benchmark${NC}"
        echo "Expected: bin/$BENCHMARK"
        exit 1
    fi
    
    echo -e "${GREEN}Build complete${NC}"
    echo ""
}

# Run profiler
run_profiler() {
    BENCHMARK_PATH="bin/$BENCHMARK"
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    
    if [ ! -f "$BENCHMARK_PATH" ]; then
        echo -e "${RED}Error: Benchmark not found: $BENCHMARK_PATH${NC}"
        exit 1
    fi
    
    case $PROFILER in
        perf)
            echo -e "${YELLOW}Running perf profiler...${NC}"
            echo "This will take a few seconds..."
            echo ""
            
            # Record profile data
            PERF_DATA="$PROFILE_DIR/perf_${BENCHMARK}_${TIMESTAMP}.data"
            
            # Run perf
            perf record -g -o "$PERF_DATA" ./$BENCHMARK_PATH 2>&1 | grep -v "^$" || true
            
            if [ ! -s "$PERF_DATA" ]; then
                echo -e "${RED}Error: perf failed to generate profile data${NC}"
                echo ""
                echo "This is usually due to kernel.perf_event_paranoid restrictions."
                echo "To enable perf for your user, run:"
                echo "  sudo sysctl -w kernel.perf_event_paranoid=1"
                echo ""
                echo "Or use gprof instead:"
                echo "  ./scripts/profile.sh $BENCHMARK gprof"
                exit 1
            fi
            
            echo ""
            echo -e "${GREEN}=== Performance Report (Top Functions) ===${NC}"
            echo ""
            
            # Generate report
            perf report -i "$PERF_DATA" --stdio -n --percent-limit 1 > "$PROFILE_DIR/perf_report_${TIMESTAMP}.txt"
            
            # Show top 20 functions
            perf report -i "$PERF_DATA" --stdio -n --percent-limit 1 | head -80
            
            echo ""
            echo -e "${BLUE}Detailed report saved to: $PROFILE_DIR/perf_report_${TIMESTAMP}.txt${NC}"
            echo -e "${BLUE}Profile data saved to: $PERF_DATA${NC}"
            echo ""
            echo "To view interactive report: perf report -i $PERF_DATA"
            ;;
            
        gprof)
            echo -e "${YELLOW}Running gprof profiler...${NC}"
            echo "This will take a few seconds..."
            echo ""
            
            # Run benchmark to generate gmon.out
            ./$BENCHMARK_PATH > /dev/null
            
            # Generate report
            GPROF_REPORT="$PROFILE_DIR/gprof_report_${TIMESTAMP}.txt"
            gprof ./$BENCHMARK_PATH gmon.out > "$GPROF_REPORT"
            
            echo ""
            echo -e "${GREEN}=== Performance Report (Top Functions) ===${NC}"
            echo ""
            
            # Show flat profile (top 20 functions by time)
            grep -A 25 "^  %   cumulative   self" "$GPROF_REPORT" | head -30
            
            echo ""
            echo -e "${BLUE}Detailed report saved to: $GPROF_REPORT${NC}"
            echo ""
            echo "To view full report: less $GPROF_REPORT"
            ;;
            
        callgrind)
            echo -e "${YELLOW}Running callgrind profiler...${NC}"
            echo "This will take 20-30 seconds (callgrind is slow)..."
            echo ""
            
            # Run with callgrind
            CALLGRIND_OUT="$PROFILE_DIR/callgrind.out.${TIMESTAMP}"
            valgrind --tool=callgrind --callgrind-out-file="$CALLGRIND_OUT" ./$BENCHMARK_PATH > /dev/null 2>&1
            
            echo ""
            echo -e "${GREEN}=== Performance Report (Top Functions) ===${NC}"
            echo ""
            
            # Generate text report
            callgrind_annotate "$CALLGRIND_OUT" > "$PROFILE_DIR/callgrind_report_${TIMESTAMP}.txt"
            
            # Show top functions
            callgrind_annotate "$CALLGRIND_OUT" | head -50
            
            echo ""
            echo -e "${BLUE}Detailed report saved to: $PROFILE_DIR/callgrind_report_${TIMESTAMP}.txt${NC}"
            echo -e "${BLUE}Callgrind data saved to: $CALLGRIND_OUT${NC}"
            echo ""
            if command -v kcachegrind &> /dev/null; then
                echo "To view interactive GUI: kcachegrind $CALLGRIND_OUT"
            else
                echo "Install kcachegrind for interactive GUI visualization"
            fi
            ;;
    esac
}



# Main execution
detect_profilers
echo -e "${GREEN}Using profiler: $PROFILER${NC}"
echo ""

build_for_profiling
run_profiler
analyze_results

echo -e "${GREEN}Profiling complete!${NC}"
