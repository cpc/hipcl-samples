#include "hip/hip_runtime.h"

__device__ float atomicAdd_SP(float *address, float val) {
  union D {
    unsigned n_uint32;
    float n_fp32;
  };

  D old, t, assumed;
  unsigned *address_as_u = (unsigned *)address;
  old.n_uint32 = *address_as_u;

  do {
    assumed.n_uint32 = old.n_uint32;
    t.n_fp32 = val + assumed.n_fp32;
    old.n_uint32 = atomicCAS(address_as_u, assumed.n_uint32, t.n_uint32);
    // Note: uses integer comparison to avoid hang in case of NaN (since NaN !=
    // NaN)
  } while (assumed.n_uint32 != old.n_uint32);

  return old.n_fp32;
}

__global__ void sum_SP(float *input, float *sum) {
  const size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
  atomicAdd_SP(sum, input[tid]);
}

int reduce_SP(int numElem) {
  // sum reduction
  size_t memSize = numElem * sizeof(float);
  int numThreads = 256;
  float *h_reduction = (float *)malloc(memSize);
  float ref_sum = 0;
  float h_sum = 0; // device to host
  for (unsigned int i = 0; i < numElem; i++) {
    h_reduction[i] = i / 2;
    ref_sum += i / 2;
  }

  float *d_reduction, *d_sum;
  hipMalloc((void **)&d_reduction, memSize);
  hipMalloc((void **)&d_sum, sizeof(float));

  // copy host memory to device to initialize to zero
  hipMemcpy(d_reduction, h_reduction, memSize, hipMemcpyHostToDevice);

  // execute the kernel
  hipLaunchKernelGGL(sum_SP, dim3(numElem / numThreads), dim3(numThreads), 0, 0,
                     d_reduction, d_sum);

  // Copy result from device to host
  hipMemcpy(&h_sum, d_sum, sizeof(float), hipMemcpyDeviceToHost);

  hipFree(d_sum);
  hipFree(d_reduction);
  free(h_reduction);

  if (h_sum != ref_sum) {
    return 1;
  }
  return 0;
}
