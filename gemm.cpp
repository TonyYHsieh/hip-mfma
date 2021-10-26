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
#include <assert.h>
#include <stdio.h>
#include <algorithm>
#include <stdlib.h>
#include<iostream>
#include "hip/hip_runtime.h"
#include "half.h"
#include "kernel.h"

////////////////////////////////////////////////////////////////////////////////////////////////////
// host code
////////////////////////////////////////////////////////////////////////////////////////////////////
using namespace std;

void deviceGemm(const half_t* hostA, const half_t* hostB, float* hostC, int M, int N, int K)
{
  // device init and run
  int lda = M;
  int ldb = N;
  int ldc = M;

  half_t* deviceA;
  half_t* deviceB;
  float*  deviceC;

  HIP_ASSERT(hipMalloc((void**)&deviceA, M*K*sizeof(half_t)));
  HIP_ASSERT(hipMalloc((void**)&deviceB, N*K*sizeof(half_t)));
  HIP_ASSERT(hipMalloc((void**)&deviceC, M*N*sizeof(float)));

  HIP_ASSERT(hipMemcpy(deviceA, hostA, M*K*sizeof(half_t), hipMemcpyHostToDevice));
  HIP_ASSERT(hipMemcpy(deviceB, hostB, N*K*sizeof(half_t), hipMemcpyHostToDevice));

  hipLaunchKernelGGL(gemmKernel,
                  dim3((M/MT0), (N/MT1)),
                  dim3(256),
                  0, 0,
                  deviceA ,deviceB ,deviceC, K, lda, ldb, ldc);

  HIP_ASSERT(hipMemcpy(hostC, deviceC, M*N*sizeof(float), hipMemcpyDeviceToHost));

  HIP_ASSERT(hipFree(deviceA));
  HIP_ASSERT(hipFree(deviceB));
  HIP_ASSERT(hipFree(deviceC));
}

void hostGemm(const half_t* a, const half_t* b, float* c, int M, int N, int K)
{
    int lda = M;
    int ldb = N;
    int ldc = M;

    for (int m=0; m<M; m++)
    {
        for (int n=0; n<N; n++)
        {
            c[m + n * ldc] = 0;
            for (int k=0; k<K; k++)
            {
                c[m + n * ldc] += (float)a[m + k * lda] * (float)b[n + k * ldb];
            }
        }
    }
}

template<typename T>
void dumpBuffer(const char* str, const T* buf, int M, int N)
{
    std::cout << "----- dump " << str << " : -----" << std::endl;
    for(int i=0; i<M; i++)
    {
        for(int j=0; j<N; j++)
        {
            std::cout << (float)buf[i] << ",";
        }
        std::cout << std::endl;
    }
}

int main(int argc, char** argv) {

  if (argc != 4)
  {
      std::cout << "a.out [m] [m] [k]" << std::endl;
  }

  int m = std::atoi(argv[1]);
  int n = std::atoi(argv[2]);
  int k = std::atoi(argv[3]);
  int errors = 0;

  std::cout << "m " << m << std::endl;
  std::cout << "n " << n << std::endl;
  std::cout << "k " << k << std::endl;

  half_t* hostA = (half_t*)malloc(m * k * sizeof(half_t));
  half_t* hostB = (half_t*)malloc(n * k * sizeof(half_t));
  float* hostC = (float*)malloc(m * n * sizeof(float));
  float* ref_C = (float*)malloc(m * n * sizeof(float));

  // initialize the input data
  for (int i = 0; i < m*k; i++) {
      hostA[i] = (half_t)(i%7-3);
  }

  for (int i = 0; i < n*k; i++) {
      hostB[i] = (half_t)(i%5-2);
  }

  for (int i = 0; i < m*n; i++) {
      hostC[i] = (float )0;
      ref_C[i] = (float )0;
  }

  // device gemm
  deviceGemm(hostA, hostB, hostC, m, n, k);

  // host gemm
  hostGemm(hostA, hostB, ref_C, m, n, k);

  hipDeviceSynchronize();

  // verify the results
  errors = 0;
  for (int i = 0; i < m * n; i++) {
      if (hostC[i] != ref_C[i]) {
          errors++;
      }
  }

  if (errors!=0) {
      printf("FAILED: %d errors\n",errors);
  } else {
      printf ("PASSED!\n");
  }

  // dumpBuffer("hostC", hostC, m, n);
  // dumpBuffer("ref_C", ref_C, m, n);

  free(hostA);
  free(hostB);
  free(hostC);
  free(ref_C);

  return errors;
}
