# Makefile for fastpf library
# Strict C90, high-performance particle filter

# ============================================================================
# Configuration
# ============================================================================

CC       = gcc
AR       = ar
# Note: -ffast-math is NOT used because it can break NaN/Inf checks and
# invalidate numerical safety assertions (e.g., detecting zero likelihood).
CFLAGS   = -std=c90 -pedantic -Wall -Wextra -Wno-long-long -O3
INCLUDES = -Iinclude
TEST_INCLUDES = -Iinclude -Itests
LDFLAGS  = -lm

# Optional OpenMP support
ifdef USE_OPENMP
    CFLAGS += -fopenmp -DFASTPF_USE_OPENMP
    LDFLAGS += -fopenmp
endif

# Debug mode
ifdef DEBUG
    CFLAGS += -g -O0 -DDEBUG
    CFLAGS := $(filter-out -O3,$(CFLAGS))
endif

# Memory sanitizers (for development/testing)
ifdef ASAN
    CFLAGS += -fsanitize=address -O1 -g -fno-omit-frame-pointer
    LDFLAGS += -fsanitize=address
    CFLAGS := $(filter-out -O3,$(CFLAGS))
endif

ifdef UBSAN
    CFLAGS += -fsanitize=undefined -O1 -g -fno-omit-frame-pointer
    LDFLAGS += -fsanitize=undefined
    CFLAGS := $(filter-out -O3,$(CFLAGS))
endif

ifdef SANITIZE
    CFLAGS += -fsanitize=address,undefined -O1 -g -fno-omit-frame-pointer
    LDFLAGS += -fsanitize=address,undefined
    CFLAGS := $(filter-out -O3,$(CFLAGS))
endif

# ============================================================================
# Directories
# ============================================================================

SRC_DIR     = src
INC_DIR     = include
BUILD_DIR   = build
BIN_DIR     = bin
TEST_DIR    = tests
UNIT_DIR    = $(TEST_DIR)/unit
ACC_DIR     = $(TEST_DIR)/accuracy
EXAMPLE_DIR = examples

# ============================================================================
# Source files
# ============================================================================

LIB_SOURCES = $(SRC_DIR)/fastpf.c \
              $(SRC_DIR)/resample.c \
              $(SRC_DIR)/rng.c \
              $(SRC_DIR)/stats.c \
              $(SRC_DIR)/util.c

LIB_OBJECTS = $(patsubst $(SRC_DIR)/%.c,$(BUILD_DIR)/%.o,$(LIB_SOURCES))

LIB_STATIC  = $(BIN_DIR)/libfastpf.a

# ============================================================================
# Test files
# ============================================================================

UNIT_TESTS = $(wildcard $(UNIT_DIR)/test_*.c)
UNIT_BINS  = $(patsubst $(UNIT_DIR)/test_%.c,$(BIN_DIR)/test_%,$(UNIT_TESTS))

ACC_TESTS  = $(wildcard $(ACC_DIR)/test_*.c)
ACC_BINS   = $(patsubst $(ACC_DIR)/test_%.c,$(BIN_DIR)/test_%,$(ACC_TESTS))

ALL_TEST_BINS = $(UNIT_BINS) $(ACC_BINS)

# ============================================================================
# Example files
# ============================================================================

EXAMPLES     = $(wildcard $(EXAMPLE_DIR)/*.c)
EXAMPLE_BINS = $(patsubst $(EXAMPLE_DIR)/%.c,$(BIN_DIR)/%,$(EXAMPLES))

# ============================================================================
# Benchmark files
# ============================================================================

BENCH_DIR     = benchmarks
BENCH_COMMON  = $(BENCH_DIR)/bench_common.c
BENCH_SOURCES = $(filter-out $(BENCH_COMMON),$(wildcard $(BENCH_DIR)/*.c))
BENCHMARK_BINS = $(patsubst $(BENCH_DIR)/%.c,$(BIN_DIR)/%,$(BENCH_SOURCES))

# ============================================================================
# Targets
# ============================================================================

.PHONY: all lib test unit accuracy examples benchmarks clean help memcheck asan ubsan sanitize

all: lib

lib: $(LIB_STATIC)

test: unit accuracy

unit: $(UNIT_BINS)
	@echo "Running unit tests..."
	@for test in $(UNIT_BINS); do \
		echo "Running $$test..."; \
		$$test || exit 1; \
	done
	@echo "All unit tests passed!"

accuracy: $(ACC_BINS)
	@echo "Running accuracy tests..."
	@for test in $(ACC_BINS); do \
		echo "Running $$test..."; \
		$$test || exit 1; \
	done
	@echo "All accuracy tests passed!"

examples: $(EXAMPLE_BINS)
	@echo "Running examples..."
	@for ex in $(EXAMPLE_BINS); do \
		echo "Running $$ex..."; \
		$$ex || exit 1; \
	done
	@echo "All examples ran successfully!"

benchmarks: $(BENCHMARK_BINS)
	@echo "Built benchmarks: $(BENCHMARK_BINS)"
	@echo "Run with: ./bin/benchmark_pf"

# ============================================================================
# Memory safety targets
# ============================================================================

memcheck: $(UNIT_BINS) $(ACC_BINS)
	@echo "=== Running Valgrind memory checks ==="
	@echo ""
	@for test in $(UNIT_BINS) $(ACC_BINS); do \
		echo "Valgrind: $$test"; \
		valgrind --leak-check=full --show-leak-kinds=all \
		         --track-origins=yes --error-exitcode=1 \
		         --suppressions=/dev/null \
		         $$test > /dev/null 2>&1 || \
		(echo "Memory issues detected in $$test" && \
		 valgrind --leak-check=full --show-leak-kinds=all \
		          --track-origins=yes $$test && exit 1); \
		echo "$$test passed Valgrind"; \
		echo ""; \
	done
	@echo "=== All tests passed Valgrind ==="

asan: clean
	@echo "=== Building with AddressSanitizer ==="
	@$(MAKE) test ASAN=1
	@echo "=== All tests passed AddressSanitizer ==="

ubsan: clean
	@echo "=== Building with UndefinedBehaviorSanitizer ==="
	@$(MAKE) test UBSAN=1
	@echo "=== All tests passed UndefinedBehaviorSanitizer ==="

sanitize: clean
	@echo "=== Building with AddressSanitizer + UndefinedBehaviorSanitizer ==="
	@$(MAKE) test SANITIZE=1
	@echo "=== All tests passed combined sanitizers ==="

clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)

help:
	@echo "fastpf Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  all       - Build library (default)"
	@echo "  lib       - Build static library"
	@echo "  test      - Run all tests (unit + accuracy)"
	@echo "  unit      - Build and run unit tests"
	@echo "  accuracy  - Build and run accuracy tests"
	@echo "  examples  - Build example programs"
	@echo "  clean     - Remove build artifacts"
	@echo ""
	@echo "Memory Safety Targets:"
	@echo "  memcheck  - Run Valgrind on all tests (requires valgrind)"
	@echo "  asan      - Build with AddressSanitizer and run tests"
	@echo "  ubsan     - Build with UndefinedBehaviorSanitizer and run tests"
	@echo "  sanitize  - Build with ASan+UBSan combined and run tests"
	@echo ""
	@echo "Options:"
	@echo "  DEBUG=1        - Build with debug symbols and no optimization"
	@echo "  USE_OPENMP=1   - Enable OpenMP parallelization"
	@echo "  ASAN=1         - Enable AddressSanitizer (for manual builds)"
	@echo "  UBSAN=1        - Enable UndefinedBehaviorSanitizer (for manual builds)"
	@echo "  SANITIZE=1     - Enable both ASan and UBSan (for manual builds)"
	@echo ""
	@echo "Example:"
	@echo "  make lib"
	@echo "  make test DEBUG=1"
	@echo "  make memcheck          # Check for memory leaks"
	@echo "  make sanitize          # Run all sanitizer checks"
	@echo "  make all USE_OPENMP=1"

# ============================================================================
# Library build rules
# ============================================================================

$(LIB_STATIC): $(LIB_OBJECTS) | $(BIN_DIR)
	$(AR) rcs $@ $^
	@echo "Built library: $@"

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

# ============================================================================
# Unit test build rules
# ============================================================================

$(BIN_DIR)/test_%: $(UNIT_DIR)/test_%.c $(LIB_STATIC) | $(BIN_DIR)
	$(CC) $(CFLAGS) $(TEST_INCLUDES) $< -L$(BIN_DIR) -lfastpf $(LDFLAGS) -o $@

# ============================================================================
# Accuracy test build rules
# ============================================================================

$(BIN_DIR)/test_%: $(ACC_DIR)/test_%.c $(LIB_STATIC) | $(BIN_DIR)
	$(CC) $(CFLAGS) $(TEST_INCLUDES) $< -L$(BIN_DIR) -lfastpf $(LDFLAGS) -o $@

# ============================================================================
# Example build rules
# ============================================================================

$(BIN_DIR)/%: $(EXAMPLE_DIR)/%.c $(LIB_STATIC) | $(BIN_DIR)
	$(CC) $(CFLAGS) $(TEST_INCLUDES) $< -L$(BIN_DIR) -lfastpf $(LDFLAGS) -o $@

# ============================================================================
# Benchmark build rules
# ============================================================================

$(BIN_DIR)/bench_%: $(BENCH_DIR)/bench_%.c $(BENCH_COMMON) $(LIB_STATIC) | $(BIN_DIR)
	$(CC) $(CFLAGS) $(TEST_INCLUDES) $< $(BENCH_COMMON) -L$(BIN_DIR) -lfastpf $(LDFLAGS) -o $@

# Old monolithic benchmark (keep for now)
$(BIN_DIR)/benchmark_pf: $(BENCH_DIR)/benchmark_pf.c $(LIB_STATIC) | $(BIN_DIR)
	$(CC) $(CFLAGS) $(TEST_INCLUDES) $< -L$(BIN_DIR) -lfastpf $(LDFLAGS) -o $@

# ============================================================================
# Directory creation
# ============================================================================

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)
