// monteCarloEngine.c
// Scott Grauer-Gray
// May 10, 2012
// Function for running Monte Carlo on the GPU using OpenCL

#include <hip/hip_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

#include "monteCarloConstants.h"

#include "monteCarloStructs.h"

#include "monteCarloKernels.h"

#include "monteCarloKernelsCpu.h"

#include "mt19937.h"

#define RISK_VAL 0.06f
#define DIV_VAL 0.0f
#define VOLT_VAL 0.200f
#define UNDERLYING_VAL 30.0f
#define STRIKE_VAL 40.0f
#define DISCOUNT_VAL 0.94176453358424872f

// initialize the inputs
void initializeInputs(dataType *samplePrices, dataType *sampleWeights,
                      dataType *times) {}

// run monte carlo...
void runMonteCarlo() {

  dataType *samplePricesGpu;
  dataType *sampleWeightsGpu;
  dataType *timesGpu;
  monteCarloOptionStruct *optionStructsGpu;
  mt19937state *randStatesGpu;

  int numSamples = 400000;
  // int numSamples = nSamplesArray[numTime];

  hipMalloc((void **)&samplePricesGpu,
            NUM_OPTIONS * numSamples * sizeof(dataType));

  hipMalloc((void **)&sampleWeightsGpu,
            NUM_OPTIONS * numSamples * sizeof(dataType));

  hipMalloc((void **)&timesGpu, NUM_OPTIONS * numSamples * sizeof(dataType));

  hipMalloc((void **)&optionStructsGpu,
            NUM_OPTIONS * sizeof(monteCarloOptionStruct));

  hipMalloc((void **)&randStatesGpu, numSamples * sizeof(mt19937state));

  printf("numSamps: %d\n", numSamples);

  // declare and initialize the struct used for the option
  monteCarloOptionStruct optionStruct;
  optionStruct.riskVal = RISK_VAL;
  optionStruct.divVal = DIV_VAL;
  optionStruct.voltVal = VOLT_VAL;
  optionStruct.underlyingVal = UNDERLYING_VAL;
  optionStruct.strikeVal = STRIKE_VAL;
  optionStruct.discountVal = DISCOUNT_VAL;

  // declare pointers for data on CPU
  dataType *samplePrices;
  dataType *sampleWeights;
  dataType *times;
  monteCarloOptionStruct *optionStructs;
  mt19937state *randStates;

  // allocate space for data on CPU
  samplePrices =
      (dataType *)malloc(NUM_OPTIONS * numSamples * sizeof(dataType));
  sampleWeights =
      (dataType *)malloc(NUM_OPTIONS * numSamples * sizeof(dataType));
  times = (dataType *)malloc(NUM_OPTIONS * numSamples * sizeof(dataType));
  optionStructs = (monteCarloOptionStruct *)malloc(
      NUM_OPTIONS * sizeof(monteCarloOptionStruct));
  randStates = (mt19937state *)malloc(numSamples * sizeof(mt19937state));

  long seconds, useconds;
  dataType mtimeGpu, mtimeCpu;
  struct timeval start;
  struct timeval end;

  for (int optNum = 0; optNum < NUM_OPTIONS; optNum++) {
    optionStructs[optNum] = optionStruct;
  }

  int samNum;

  // transfer data to device
  hipMemcpy(samplePricesGpu, samplePrices,
            NUM_OPTIONS * numSamples * sizeof(dataType), hipMemcpyHostToDevice);
  hipMemcpy(sampleWeightsGpu, sampleWeights,
            NUM_OPTIONS * numSamples * sizeof(dataType), hipMemcpyHostToDevice);
  hipMemcpy(timesGpu, times, NUM_OPTIONS * numSamples * sizeof(dataType),
            hipMemcpyHostToDevice);
  hipMemcpy(optionStructsGpu, optionStructs,
            NUM_OPTIONS * sizeof(monteCarloOptionStruct),
            hipMemcpyHostToDevice);
  hipMemcpy(randStatesGpu, randStates, numSamples * sizeof(mt19937state),
            hipMemcpyHostToDevice);

  dataType dt = (1.0f / (dataType)SEQUENCE_LENGTH);

  unsigned int timer = 0;

  srand(time(NULL));

  gettimeofday(&start, NULL);

  /* initialize random seed: */
  srand(rand());

  printf("\nRun on GPU\n");

  size_t localWorkSize = 256;
  size_t globalWorkSize = ceil((dataType)numSamples / (dataType)localWorkSize);

  unsigned long seed = rand();

  hipLaunchKernelGGL(initializeMersenneStateGpu, dim3(globalWorkSize),
                     dim3(localWorkSize), 0, 0, randStatesGpu, seed,
                     numSamples);

  hipLaunchKernelGGL(monteCarloGpuKernel, dim3(globalWorkSize),
                     dim3(localWorkSize), 0, 0, samplePricesGpu,
                     sampleWeightsGpu, timesGpu, dt, randStatesGpu,
                     optionStructsGpu, numSamples);

  // transfer data back to host

  hipMemcpy(samplePrices, samplePricesGpu, numSamples * sizeof(dataType),
            hipMemcpyDeviceToHost);

  // retrieve the average price
  dataType cumPrice = 0.0f;

  // add all the computed prices together
  for (int numSamp = 0; numSamp < numSamples; numSamp++) {
    cumPrice += samplePrices[numSamp];
  }

  dataType avgPrice = cumPrice / numSamples;

  gettimeofday(&end, NULL);

  printf("Average price on GPU: %f\n", avgPrice);

  seconds = end.tv_sec - start.tv_sec;
  useconds = end.tv_usec - start.tv_usec;

  mtimeGpu = ((seconds)*1000 + ((dataType)useconds) / 1000.0) + 0.5;

  printf("Processing time on GPU: %f (ms)\n\n", mtimeGpu);

  // free memory space on the GPU
  hipFree(samplePricesGpu);
  hipFree(sampleWeightsGpu);
  hipFree(timesGpu);
  hipFree(optionStructsGpu);
  hipFree(randStatesGpu);

  // free memory space on the CPU
  free(samplePrices);
  free(sampleWeights);
  free(times);

  // declare pointers for data on CPU
  dataType *samplePricesCpu;
  dataType *sampleWeightsCpu;
  dataType *timesCpu;

  // allocate space for data on CPU
  samplePricesCpu = (dataType *)malloc(numSamples * sizeof(dataType));
  sampleWeightsCpu = (dataType *)malloc(numSamples * sizeof(dataType));
  timesCpu = (dataType *)malloc(numSamples * sizeof(dataType));

  printf("Run on CPU\n");

  gettimeofday(&start, NULL);

  monteCarloGpuKernelCpu(samplePricesCpu, sampleWeightsCpu, timesCpu,
                         (1.0f / (dataType)SEQUENCE_LENGTH), optionStructs,
                         numSamples);

  cumPrice = 0.0f;
  // add all the computed prices together
  for (int numSamp = 0; numSamp < numSamples; numSamp++) {

    cumPrice += samplePricesCpu[numSamp];
  }

  avgPrice = cumPrice / numSamples;

  gettimeofday(&end, NULL);

  seconds = end.tv_sec - start.tv_sec;
  useconds = end.tv_usec - start.tv_usec;

  mtimeCpu = ((seconds)*1000 + ((dataType)useconds) / 1000.0) + 0.5;
  printf("Processing time on CPU: %f (ms)\n", mtimeCpu);

  // retrieve the average price
  cumPrice = 0.0f;

  printf("Average price on CPU: %f\n\n", avgPrice);

  printf("GPU Speedup: %f\n", mtimeCpu / mtimeGpu);

  // free memory space on the CPU
  free(samplePricesCpu);
  free(sampleWeightsCpu);
  free(timesCpu);
  free(optionStructs);
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
  runMonteCarlo();

  return 0;
}
