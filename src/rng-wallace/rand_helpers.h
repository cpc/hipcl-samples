// ************************************************
// rand_helpers.h
// authors: Lee Howes and David B. Thomas
//
// Contains support code for the random number
// generation necessary for initialising the
// cuda simulations correctly.
//
// Ziggurat code taken from Marsaglia's
// paper.
// ************************************************

#ifndef __rand_helpers_h
#define __rand_helpers_h

void computeRNG ();

extern float *devPool;
extern float *hostPool;

unsigned Kiss ();
double Rand ();

void initRand ();
double RandN ();
void init_genrand (unsigned long s);
void init_by_array (unsigned long init_key[], int key_length);
unsigned long genrand_int32 (void);
double genrand_real2 (void);
double genrand_real3 (void);

float nfix (void);

void initRand ();

double MakeChi2Scale (unsigned N);

#endif
