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

////////////////////////////////////////////////////////////////////////////////////////////////////
// host code
////////////////////////////////////////////////////////////////////////////////////////////////////
using namespace std;

extern float deviceGemm(const half_t* hostA, const half_t* hostB, float* hostC, int M, int N, int K, int Iteration);

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

  if (argc != 6)
  {
      std::cout << "a.out [m] [m] [k] [v] [iter]" << std::endl;
  }

  int m    = std::atoi(argv[1]);
  int n    = std::atoi(argv[2]);
  int k    = std::atoi(argv[3]);
  int val  = std::atoi(argv[4]);
  int iter = std::atoi(argv[5]);

  int errors = 0;


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
  float gflops = deviceGemm(hostA, hostB, hostC, m, n, k, iter);

  if (val)
  {
      // host gemm
      hostGemm(hostA, hostB, ref_C, m, n, k);

      // verify the results
      errors = 0;
      for (int i = 0; i < m * n; i++) {
          if (hostC[i] != ref_C[i]) {
              errors++;
          }
      }
  }

  // conclusion
  std::cout << " m " << m << " n " << n << " k " << k << " gflops " << gflops << " ";
  if (val)
  {
     if (errors)
         std::cout << "Fail:" << errors;
     else
         std::cout << "Pass!";
  }
  std::cout << std::endl;

  free(hostA);
  free(hostB);
  free(hostC);
  free(ref_C);

  return errors;
}
