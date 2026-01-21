/**
 * @file test_common.h
 * @brief Common test utilities and C90 compatibility macros.
 */

#ifndef TEST_COMMON_H
#define TEST_COMMON_H

#include <stdio.h>

/* Include internal definitions for testing (allows stack allocation) */
#include "../src/fastpf_internal.h"

/* C90 compatibility for INFINITY */
#ifndef INFINITY
    #define INFINITY (1.0 / 0.0)
#endif

/* C90 compatibility for isnan/isinf */
#ifndef isnan
    #define isnan(x) ((x) != (x))
#endif
#ifndef isinf
    #define isinf(x) (!isnan(x) && isnan((x) - (x)))
#endif

/* Test helper macros */
#define TEST_ASSERT(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL: %s\n", msg); \
        return 1; \
    } \
} while(0)

#define TEST_PASS(name) do { \
    printf("PASS: %s\n", name); \
} while(0)

#endif /* TEST_COMMON_H */
