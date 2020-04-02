
#ifndef _RNG_KERNEL_CU_
#define _RNG_KERNEL_CU_

#include <hip/hip_runtime.h>

#include "constants.h"
#include "rand_helpers.h"

float *randomNumbers;
float *device_randomNumbers;

float *rngChi2Corrections = 0;
float *devicerngChi2Corrections = 0;

__global__ void rng_wallace(unsigned seed, float *globalPool,
                            float *generatedRandomNumberPool,
                            float *chi2Corrections);

void init_rng_tests() {

  rngChi2Corrections = (float *)malloc(4 * WALLACE_CHI2_COUNT);

  randomNumbers = (float *)malloc(4 * WALLACE_OUTPUT_SIZE);

  // Asian option memory allocations
  (hipMalloc((void **)&devicerngChi2Corrections, 4 * WALLACE_CHI2_COUNT));

  (hipMalloc((void **)&device_randomNumbers, 4 * WALLACE_OUTPUT_SIZE));
  // Initialise asian option parameters, random guesses at this point...

  for (int i = 0; i < WALLACE_CHI2_COUNT; i++) {
    rngChi2Corrections[i] = MakeChi2Scale(WALLACE_TOTAL_POOL_SIZE);
  }
  (hipMemcpy(devicerngChi2Corrections, rngChi2Corrections,
             4 * WALLACE_CHI2_COUNT, hipMemcpyHostToDevice));
}

void cleanup_rng_options() {

  free(rngChi2Corrections);

  free(randomNumbers);

  (hipFree(devicerngChi2Corrections));

  (hipFree(device_randomNumbers));
}

void computeRNG() {

  init_rng_tests();

  // setup execution parameters and execute
  dim3 rng_wallace_grid(WALLACE_NUM_BLOCKS, 1, 1);
  dim3 rng_wallace_threads(WALLACE_NUM_THREADS, 1, 1);

  // Execute the Tausworthe RNG, outputting into memory, and timing as we go.
  unsigned seed = 1;

  // Execute the Wallace RNG, outputting into memory, and timing as we go.
  hipLaunchKernelGGL(
      rng_wallace, dim3(rng_wallace_grid), dim3(rng_wallace_threads),
      4 * (WALLACE_POOL_SIZE + WALLACE_CHI2_SHARED_SIZE), 0, seed, devPool,
      device_randomNumbers, devicerngChi2Corrections);

  hipMemcpy(randomNumbers, device_randomNumbers, 4 * WALLACE_OUTPUT_SIZE,
            hipMemcpyDeviceToHost);

  // verification
  /*
          for (int i = 0; i < WALLACE_OUTPUT_SIZE; i++)
                  printf("%.3f\n", randomNumbers[i]);
  */
  cleanup_rng_options();
}

#endif
