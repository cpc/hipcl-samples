/**
 * main-hip.cpp: This file is part of the mixbench GPU micro-benchmark suite.
 *
 * Contact: Elias Konstantinidis <ekondis@gmail.com>
 **/

#include "lhiputil.h"
#include "mix_kernels_hip.h"
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define VECTOR_SIZE (8 * 1024 * 1024)

void init_vector(double *v, size_t datasize) {
  for (int i = 0; i < (int)datasize; i++)
    v[i] = i;
}

int main(int argc, char *argv[]) {
  printf("mixbench-hip/alternating (compute & memory balancing GPU "
         "microbenchmark)\n");

  unsigned int datasize = VECTOR_SIZE * sizeof(double);

  hipSetDevice(0);
  StoreDeviceInfo(stdout);

  size_t freeCUDAMem, totalCUDAMem;
  hipMemGetInfo(&freeCUDAMem, &totalCUDAMem);
  printf("Total GPU memory %lu, free %lu\n", totalCUDAMem, freeCUDAMem);
  printf("Buffer size: %dMB\n", datasize / (1024 * 1024));

  double *c;
  c = (double *)malloc(datasize);
  init_vector(c, VECTOR_SIZE);

  mixbenchGPU(c, VECTOR_SIZE);

  free(c);

  return 0;
}
