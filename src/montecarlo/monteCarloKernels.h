// monteCarloKernels.cuh
// Scott Grauer-Gray
// May 10, 2012
// Kernel headers for running monte carlo on the GPU

#ifndef MONTE_CARLO_KERNELS_CUH
#define MONTE_CARLO_KERNELS_CUH

// needed for constants related to monte carlo
#include "monteCarloConstants.h"

#include "mt19937.h"

// needed for structs related to monte carlo
#include "monteCarloStructs.h"

// function to compute the inverse normal distribution
__device__ dataType compInverseNormDist(dataType x);

__device__ dataType interestRateCompoundFact(dataType t, dataType rate);

__device__ dataType interestRateDiscountFact(dataType t, dataType rate);

__device__ dataType flatForwardDiscountImpl(dataType t, dataType rate);

__device__ dataType yieldTermStructDiscount(dataType t, dataType rate);

__device__ dataType interestRateImpliedRate(dataType compound, dataType t);

__device__ dataType yieldTermStructForwardRate(dataType t1, dataType t2,
                                               dataType rate);

__device__ dataType processDrift(dataType t, dataType x,
                                 monteCarloOptionStruct optionStruct);

__device__ dataType discretizationDrift(dataType t0, dataType x0, dataType dt,
                                        monteCarloOptionStruct optionStruct);

__device__ dataType localVoltLocVol(dataType t, dataType underlyingLevel,
                                    monteCarloOptionStruct optionStruct);

__device__ dataType processDiff(dataType t, dataType x,
                                monteCarloOptionStruct optionStruct);

__device__ dataType discDiff(dataType t0, dataType x0, dataType dt,
                             monteCarloOptionStruct optionStruct);

__device__ dataType stdDeviation(dataType t0, dataType x0, dataType dt,
                                 monteCarloOptionStruct optionStruct);

__device__ dataType apply(dataType x0, dataType dx);

__device__ dataType discDrift(dataType t0, dataType x0, dataType dt,
                              monteCarloOptionStruct optionStruct);

__device__ dataType processEvolve(dataType t0, dataType x0, dataType dt,
                                  dataType dw,
                                  monteCarloOptionStruct optionStruct);

// retrieve the current sequence
__device__ void getSequence(dataType *sequence, dataType sampleNum);

__device__ dataType getProcessValX0(monteCarloOptionStruct optionStruct);

__device__ void getPath(dataType *path, size_t sampleNum, dataType dt,
                        mt19937state *state,
                        monteCarloOptionStruct optionStruct);

__device__ dataType getPrice(dataType val);

// initialize the path
__device__ void initializePath(dataType *path);

__global__ void monteCarloGpuKernel(dataType *samplePrices,
                                    dataType *sampleWeights, dataType *times,
                                    dataType dt, mt19937state *randStates,
                                    monteCarloOptionStruct *optionStructs,
                                    int numSamples);

__global__ void initializeMersenneStateGpu(mt19937state *m, unsigned long seed,
                                           int numSamples);

#endif // MONTE_CARLO_KERNELS_CUH
