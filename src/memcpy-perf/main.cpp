/*
Copyright (c) 2015-2016 Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <iostream>
#include <cstdlib>
#include <time.h>
#include "hip/hip_runtime.h"

#define NUM_SIZE 16
#define NUM_ITER (1 << 16)

void setup(size_t *size) {
    for (int i = 0; i < NUM_SIZE; i++) {
        size[i] = 1 << (i + 6);  // start at 8 bytes
    }
}

void valSet(int* A, int val, size_t size) {
    size_t len = size / sizeof(int);
    for (int i = 0; i < len; i++) {
        A[i] = val;
    }
}

int main() {
    int *A, *Ad;
    size_t size[NUM_SIZE];
    hipError_t err;

    setup(size);
    for (int i = 0; i < NUM_SIZE; i++) {
        A = (int*)malloc(size[i]);
	if (A == nullptr) {
          std::cerr << "Host memory allocation failed\n";
	  return -1;
	}	
        valSet(A, 1, size[i]);
        err = hipMalloc((void**)&Ad, size[i]);
	if (err != hipSuccess) {
          std::cerr << "Device memory allocation failed\n";
	  free(A);
	  return -1;
	}
        clock_t start, end;
        start = clock();
        for (int j = 0; j < NUM_ITER; j++) {
            hipMemcpy(Ad, A, size[i], hipMemcpyHostToDevice);
        }
        hipDeviceSynchronize();
        end = clock();
        double uS = (double)(end - start) * 1000 / (NUM_ITER * CLOCKS_PER_SEC);
        std::cout << "Copy " << size[i] << " btyes from host to device takes " 
		  << uS <<  " us" << std::endl;
        hipFree(Ad);
        free(A);
    }
    return 0;
}
