/**
 * @file rng.c
 * @brief PCG32 random number generator with Ziggurat normal sampling.
 *
 * Based on:
 * - PCG family by Melissa O'Neill (www.pcg-random.org)
 * - Ziggurat algorithm by Marsaglia & Tsang (2000)
 *
 * The Ziggurat method is the industry standard for fast, exact normal sampling
 * (used in NumPy, R, Matlab). It achieves ~3× speedup over Box-Muller by using
 * rejection sampling with precomputed lookup tables.
 */

#include "fastpf.h"
#include "fastpf_internal.h"
#include <math.h>

/* PCG32 constants */
#define PCG32_MULT 6364136223846793005ULL
#define PCG32_INCREMENT 1442695040888963407ULL

/* Ziggurat method constants for 128 layers (Marsaglia & Tsang). */
#define ZIGGURAT_N 128
#define ZIGGURAT_R 3.442619855899
#define ZIGGURAT_INV_R 0.29047645161474336
#define ZIGGURAT_V 9.91256303526217e-3

/* Ziggurat tables computed at runtime (once) from constants above. */
static unsigned long ziggurat_kn[ZIGGURAT_N];
static double ziggurat_wn[ZIGGURAT_N];
static double ziggurat_fn[ZIGGURAT_N];
static int ziggurat_initialized = 0;

static void ziggurat_init(void)
{
    int i;
    double dn, tn, vn, q;
    double m1;

    if (ziggurat_initialized) {
        return;
    }

#ifdef FASTPF_USE_OPENMP
    #pragma omp critical
    {
        if (ziggurat_initialized) {
            return;
        }
#endif
        m1 = 2147483648.0; /* 2^31 */
        dn = ZIGGURAT_R;
        tn = dn;
        vn = ZIGGURAT_V;

        q = vn / exp(-0.5 * dn * dn);
        ziggurat_kn[0] = (unsigned long)((dn / q) * m1);
        ziggurat_kn[1] = 0;    /* Canonical zigset init: layer 1 always uses slow path */

        ziggurat_wn[0] = q / m1;
        ziggurat_wn[ZIGGURAT_N - 1] = dn / m1;

        ziggurat_fn[0] = 1.0;
        ziggurat_fn[ZIGGURAT_N - 1] = exp(-0.5 * dn * dn);

        for (i = ZIGGURAT_N - 2; i >= 1; i--) {
            dn = sqrt(-2.0 * log(vn / dn + exp(-0.5 * dn * dn)));
            ziggurat_kn[i + 1] = (unsigned long)((dn / tn) * m1);
            tn = dn;
            ziggurat_fn[i] = exp(-0.5 * dn * dn);
            ziggurat_wn[i] = dn / m1;
        }

        ziggurat_initialized = 1;
#ifdef FASTPF_USE_OPENMP
    }
#endif
}

/* Tail sampling for |x| > R using exponential distribution */
static double ziggurat_sample_tail(fastpf_rng_t* rng) {
    double x, y;
    do {
        x = -log(fastpf_rng_uniform01(rng)) * ZIGGURAT_INV_R;
        y = -log(fastpf_rng_uniform01(rng));
    } while (y + y < x * x);
    return ZIGGURAT_R + x;
}

/* ========================================================================
 * PCG32 implementation
 * ======================================================================== */

/**
 * @brief Seed the PCG32 RNG with a 64-bit seed.
 *
 * Initializes the internal state and stream selector, then advances twice
 * to diffuse the seed into the state (standard PCG seeding sequence).
 *
 * @param rng Pointer to RNG state (must be valid).
 * @param seed 64-bit seed value.
 */
void fastpf_rng_seed(fastpf_rng_t* rng, uint64_t seed)
{
    rng->state = 0ULL;
    rng->inc = (PCG32_INCREMENT << 1u) | 1u;
    fastpf_rng_u32(rng); /* Advance once */
    rng->state += seed;
    fastpf_rng_u32(rng); /* Advance again */
}

/**
 * @brief Generate a uniformly distributed 32-bit unsigned integer.
 *
 * Uses the PCG32 output permutation (XSH RR) to produce high-quality
 * 32-bit values from the 64-bit internal state.
 *
 * @param rng Pointer to RNG state (must be initialized).
 * @return 32-bit random value in [0, 2^32 - 1].
 */
unsigned int fastpf_rng_u32(fastpf_rng_t* rng)
{
    uint64_t oldstate;
    unsigned int xorshifted, rot;

    oldstate = rng->state;
    rng->state = oldstate * PCG32_MULT + rng->inc;
    
    xorshifted = (unsigned int)(((oldstate >> 18u) ^ oldstate) >> 27u);
    rot = (unsigned int)(oldstate >> 59u);
    
    return (xorshifted >> rot) | (xorshifted << ((-(int)rot) & 31));
}

/**
 * @brief Generate a uniform double in the open interval (0, 1).
 *
 * Uses a 32-bit draw and maps to (0,1) by adding 1 and dividing by 2^32 + 1,
 * which avoids exact 0 and 1 (important for log and Box-Muller).
 *
 * @param rng Pointer to RNG state (must be initialized).
 * @return Uniform random double in (0, 1).
 */
double fastpf_rng_uniform01(fastpf_rng_t* rng)
{
    unsigned int u;
    double result;
    
    u = fastpf_rng_u32(rng);
    result = (u + 1.0) / 4294967297.0; /* (2^32 + 1) */
    
    return result;
}

/**
 * @brief Generate a standard normal N(0,1) variate using Ziggurat method.
 *
 * The Ziggurat algorithm (Marsaglia & Tsang, 2000) provides ~3× speedup
 * over Box-Muller by using rejection sampling with precomputed tables.
 *
 * Algorithm:
 * 1. Pick random layer i and x-coordinate
 * 2. If x < x[i], accept immediately (98.9% of cases)
 * 3. Otherwise, expensive check or tail sampling (1.1% of cases)
 *
 * @param rng Pointer to RNG state (must be initialized).
 * @return Standard normal random value.
 */
double fastpf_rng_normal(fastpf_rng_t* rng)
{
    unsigned int u;
    unsigned int iz;
    unsigned int uabs;
    double x, y;

    ziggurat_init();

    while (1) {
        /* Pick random layer and generate candidate x */
        u = fastpf_rng_u32(rng);
        iz = u & (ZIGGURAT_N - 1U);
        uabs = u & 0x7fffffffU;
        
        x = (double)uabs * ziggurat_wn[iz];
        
        /* Fast path: accept if within rectangle */
        if (uabs < ziggurat_kn[iz]) {
            return (u & 0x80000000U) ? -x : x;
        }
        
        /* Layer 0: tail sampling */
        if (iz == 0) {
            x = ziggurat_sample_tail(rng);
            return (u & 0x80000000U) ? -x : x;
        }
        
        /* Slow path: exact acceptance check */
        y = ziggurat_fn[iz] + fastpf_rng_uniform01(rng) *
            (ziggurat_fn[iz - 1] - ziggurat_fn[iz]);
        
        if (y < exp(-0.5 * x * x)) {
            return (u & 0x80000000U) ? -x : x;
        }
        
        /* Rejection: try again */
    }
}
