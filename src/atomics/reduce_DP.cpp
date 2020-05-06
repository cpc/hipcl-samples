#include "hip/hip_runtime.h"

__device__ double atomicAdd_DP(double *address, double val) {
  union D {
    unsigned long long int n_uint64;
    double n_fp64;
  };

  D old, t, assumed;
  unsigned long long int *address_as_ul = (unsigned long long int *)address;
  old.n_uint64 = *address_as_ul;

  do {
    assumed.n_uint64 = old.n_uint64;
    t.n_fp64 = val + assumed.n_fp64;
    old.n_uint64 = atomicCAS(address_as_ul, assumed.n_uint64, t.n_uint64);
    // Note: uses integer comparison to avoid hang in case of NaN (since NaN !=
    // NaN)
  } while (assumed.n_uint64 != old.n_uint64);

  return old.n_fp64;
}

__global__ void sum_DP(double *input, double *sum) {
  const size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
  atomicAdd_DP(sum, input[tid]);
}

int reduce_DP(int numElem) {
  // sum reduction
  size_t memSize = numElem * sizeof(double);
  int numThreads = 256;
  double *h_reduction = (double *)malloc(memSize);
  double ref_sum = 0.0;
  double h_sum = 0.0; // device to host
  for (unsigned int i = 0; i < numElem; i++) {
    h_reduction[i] = i / 2;
    ref_sum += i / 2;
  }

  double *d_reduction, *d_sum;
  hipMalloc((void **)&d_reduction, memSize);
  hipMalloc((void **)&d_sum, sizeof(double));

  // copy host memory to device to initialize to zero
  hipMemcpy(d_reduction, h_reduction, memSize, hipMemcpyHostToDevice);
  hipMemcpy(d_sum, &h_sum, sizeof(double), hipMemcpyHostToDevice);

  // execute the kernel
  hipLaunchKernelGGL(sum_DP, dim3(numElem / numThreads), dim3(numThreads), 0, 0,
                     d_reduction, d_sum);

  // Copy result from device to host
  hipMemcpy(&h_sum, d_sum, sizeof(double), hipMemcpyDeviceToHost);

  hipDeviceSynchronize();

  hipFree(d_sum);
  hipFree(d_reduction);
  free(h_reduction);

  if (h_sum != ref_sum) {
    return 1;
  }
  return 0;
}
