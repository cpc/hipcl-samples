/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

// System includes
#include <assert.h>
#include <stdio.h>
#include <hip/hip_runtime.h>


#ifndef MAX
#define MAX(a, b) (a > b ? a : b)
#endif

////////////////////////////////////////////////////////////////////////////////
// Global types and parameters
////////////////////////////////////////////////////////////////////////////////
#define VOTE_DATA_GROUP 4

////////////////////////////////////////////////////////////////////////////////
// Voting Kernel functions
////////////////////////////////////////////////////////////////////////////////
#include "vote_kernel.h"

// Generate the test pattern for Tests 1 and 2
void genVoteTestPattern(unsigned int *VOTE_PATTERN, int size) {
  // For testing VOTE.Any (all of these threads will return 0)
  for (int i = 0; i < size / 4; i++) {
    VOTE_PATTERN[i] = 0x00000000;
  }

  // For testing VOTE.Any (1/2 these threads will return 1)
  for (int i = 2 * size / 8; i < 4 * size / 8; i++) {
    VOTE_PATTERN[i] = (i & 0x01) ? i : 0;
  }

  // For testing VOTE.all (1/2 of these threads will return 0)
  for (int i = 2 * size / 4; i < 3 * size / 4; i++) {
    VOTE_PATTERN[i] = (i & 0x01) ? 0 : i;
  }

  // For testing VOTE.all (all of these threads will return 1)
  for (int i = 3 * size / 4; i < size; i++) {
    VOTE_PATTERN[i] = 0xffffffff;
  }
}

int checkErrors1(unsigned int *h_result, int start, int end, int warp_size,
                 const char *voteType) {
  int i, sum = 0;

  for (i = start; i < end; i++) {
    sum += h_result[i];
  }

  if (sum > 0) {
    printf("\t<%s>[%d - %d] = ", voteType, start, end - 1);

    for (i = start; i < end; i++) {
      printf("%d", h_result[i]);
    }

    printf("%d values FAILED\n", sum);
  }

  return (sum > 0);
}

int checkErrors2(unsigned int *h_result, int start, int end, int warp_size,
                 const char *voteType) {
  int i, sum = 0;

  for (i = start; i < end; i++) {
    sum += h_result[i];
  }

  if (sum != warp_size) {
    printf("\t<%s>[%d - %d] = ", voteType, start, end - 1);

    for (i = start; i < end; i++) {
      printf("%d", h_result[i]);
    }

    printf(" - FAILED\n");
  }

  return (sum != warp_size);
}

// Verification code for Kernel #1
int checkResultsVoteAnyKernel1(unsigned int *h_result, int size,
                               int warp_size) {
  int error_count = 0;

  error_count += checkErrors1(h_result, 0, VOTE_DATA_GROUP * warp_size / 4,
                              warp_size, "Vote.Any");
  error_count +=
      checkErrors2(h_result, VOTE_DATA_GROUP * warp_size / 4,
                   2 * VOTE_DATA_GROUP * warp_size / 4, warp_size, "Vote.Any");
  error_count +=
      checkErrors2(h_result, 2 * VOTE_DATA_GROUP * warp_size / 4,
                   3 * VOTE_DATA_GROUP * warp_size / 4, warp_size, "Vote.Any");
  error_count +=
      checkErrors2(h_result, 3 * VOTE_DATA_GROUP * warp_size / 4,
                   4 * VOTE_DATA_GROUP * warp_size / 4, warp_size, "Vote.Any");

  printf((error_count == 0) ? "\tOK\n" : "\tERROR\n");
  return error_count;
}

// Verification code for Kernel #2
int checkResultsVoteAllKernel2(unsigned int *h_result, int size,
                               int warp_size) {
  int error_count = 0;

  error_count += checkErrors1(h_result, 0, VOTE_DATA_GROUP * warp_size / 4,
                              warp_size, "Vote.All");
  error_count +=
      checkErrors1(h_result, VOTE_DATA_GROUP * warp_size / 4,
                   2 * VOTE_DATA_GROUP * warp_size / 4, warp_size, "Vote.All");
  error_count +=
      checkErrors1(h_result, 2 * VOTE_DATA_GROUP * warp_size / 4,
                   3 * VOTE_DATA_GROUP * warp_size / 4, warp_size, "Vote.All");
  error_count +=
      checkErrors2(h_result, 3 * VOTE_DATA_GROUP * warp_size / 4,
                   4 * VOTE_DATA_GROUP * warp_size / 4, warp_size, "Vote.All");

  printf((error_count == 0) ? "\tOK\n" : "\tERROR\n");
  return error_count;
}

// Verification code for Kernel #3
int checkResultsVoteAnyKernel3(bool *hinfo, int size) {
  int i, error_count = 0;

  for (i = 0; i < size * 3; i++) {
    switch (i % 3) {
      case 0:

        // First warp should be all zeros.
        if (hinfo[i] != (i >= size * 1)) {
          error_count++;
        }

        break;

      case 1:

        // First warp and half of second should be all zeros.
        if (hinfo[i] != (i >= size * 3 / 2)) {
          error_count++;
        }

        break;

      case 2:

        // First two warps should be all zeros.
        if (hinfo[i] != (i >= size * 2)) {
          error_count++;
        }

        break;
    }
  }

  printf((error_count == 0) ? "\tOK\n" : "\tERROR\n");
  return error_count;
}

// We assume that the kernels are executed on a GPU.
int main(int argc, char **argv) {
  unsigned int *h_input, *h_result;
  unsigned int *d_input, *d_result;

  bool *dinfo = NULL, *hinfo = NULL;
  int error_count[3] = {0, 0, 0};

  hipDeviceProp_t deviceProp;
  int devID; 
      
  int warp_size = 32; 
  //  default warp size may be set as a command-line argument
  if (argc == 2) {
    warp_size = atoi(argv[1]);
  }

  // This will pick the default device, but it must be a GPU
  devID = 0;
  hipGetDeviceProperties(&deviceProp, devID);

  // Statistics about the GPU device
  printf("> Device Name %s\n", deviceProp.name);
  printf("> Device has %d Multi-Processors\n\n", deviceProp.multiProcessorCount);

  h_input = (unsigned int *)malloc(VOTE_DATA_GROUP * warp_size *
                                   sizeof(unsigned int));
  h_result = (unsigned int *)malloc(VOTE_DATA_GROUP * warp_size *
                                    sizeof(unsigned int));
  hipMalloc((void**)&d_input, VOTE_DATA_GROUP * warp_size * sizeof(unsigned int));
  hipMalloc((void**)&d_result, VOTE_DATA_GROUP * warp_size * sizeof(unsigned int));
  genVoteTestPattern(h_input, VOTE_DATA_GROUP * warp_size);
  hipMemcpy(d_input, h_input, 
      VOTE_DATA_GROUP * warp_size * sizeof(unsigned int), hipMemcpyHostToDevice);

  // Start of Vote Any Test Kernel #1
  printf("[VOTE Kernel Test 1/3]\n");
  printf("\tRunning <<Vote.Any>> kernel1 ...\n");
  {
    hipDeviceSynchronize();
    dim3 gridBlock(1, 1);
    dim3 threadBlock(VOTE_DATA_GROUP * warp_size, 1);
    hipLaunchKernelGGL(VoteAnyKernel1, dim3(gridBlock), dim3(threadBlock), 0, 0, d_input, d_result,
                                               VOTE_DATA_GROUP * warp_size);
    hipDeviceSynchronize();
  }
  hipMemcpy(h_result, d_result,
                             VOTE_DATA_GROUP * warp_size * sizeof(unsigned int),
                             hipMemcpyDeviceToHost);
  error_count[0] += checkResultsVoteAnyKernel1(
      h_result, VOTE_DATA_GROUP * warp_size, warp_size);

  // Start of Vote All Test Kernel #2
  printf("\n[VOTE Kernel Test 2/3]\n");
  printf("\tRunning <<Vote.All>> kernel2 ...\n");
  {
    hipDeviceSynchronize();
    dim3 gridBlock(1, 1);
    dim3 threadBlock(VOTE_DATA_GROUP * warp_size, 1);
    hipLaunchKernelGGL(VoteAllKernel2, dim3(gridBlock), dim3(threadBlock), 0, 0, d_input, d_result,
                                               VOTE_DATA_GROUP * warp_size);
    hipDeviceSynchronize();
  }
  hipMemcpy(h_result, d_result,
             VOTE_DATA_GROUP * warp_size * sizeof(unsigned int),
             hipMemcpyDeviceToHost);

  error_count[1] += checkResultsVoteAllKernel2(
      h_result, VOTE_DATA_GROUP * warp_size, warp_size);

  // Second Vote Kernel Test #3 (both Any/All)
  hinfo = reinterpret_cast<bool *>(calloc(warp_size * 3 * 3, sizeof(bool)));
  hipMalloc((void **)&dinfo, warp_size * 3 * 3 * sizeof(bool));
  hipMemcpy(dinfo, hinfo, warp_size * 3 * 3 * sizeof(bool),
             hipMemcpyHostToDevice);

  printf("\n[VOTE Kernel Test 3/3]\n");
  printf("\tRunning <<Vote.Any>> kernel3 ...\n");
  {
    hipDeviceSynchronize();
    hipLaunchKernelGGL(VoteAnyKernel3, dim3(1), dim3(warp_size * 3), 0, 0, dinfo, warp_size);
    hipDeviceSynchronize();
  }

  hipMemcpy(hinfo, dinfo, warp_size * 3 * 3 * sizeof(bool),
             hipMemcpyDeviceToHost);

  error_count[2] = checkResultsVoteAnyKernel3(hinfo, warp_size * 3);

  // Now free these resources for Test #1,2
  hipFree(d_input);
  hipFree(d_result);
  free(h_input);
  free(h_result);

  // Free resources from Test #3
  free(hinfo);
  hipFree(dinfo);

  return (error_count[0] == 0 && error_count[1] == 0 && error_count[2] == 0)
             ? EXIT_SUCCESS
             : EXIT_FAILURE;
}
