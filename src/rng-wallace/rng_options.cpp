#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hip/hip_runtime.h"
#include "constants.h"
#include "rand_helpers.h"

float *devPool = 0;
float *hostPool = 0;

int main() {
  hostPool = (float *)malloc(4 * WALLACE_TOTAL_POOL_SIZE);
  for (unsigned i = 0; i < WALLACE_TOTAL_POOL_SIZE; i++) {
    float x = RandN();
    hostPool[i] = x;
  }
  (hipMalloc((void **)&devPool, 4 * WALLACE_TOTAL_POOL_SIZE));
  (hipMemcpy(devPool, hostPool, 4 * WALLACE_TOTAL_POOL_SIZE,
             hipMemcpyHostToDevice));

  computeRNG();

  (hipFree(devPool));
  free(hostPool);
  return 0;
}
