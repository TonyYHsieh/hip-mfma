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

#define HIP_ASSERT(x) (assert((x)==hipSuccess))

#define WAVE_SIZE 64

#define MFMA_T 32
#define MFMA_K  8

#define MT0 256  // Macro tile
#define MT1 128

#define WT0 128  // wrap tile
#define WT1 64

#define TT0 4 // thread tile
#define TT1 2

#define DepthU 8

#define MIIN 4 // inputer per mi instruction

#define MFMA_OU 4
#define MFMA_OG 4

// Native types can use explicit vector extension
template <typename T, int Elements>
struct VectorStorage
{
    using type = T __attribute__((ext_vector_type(Elements)));
};

__global__ __launch_bounds__(256, 2)
void gemmKernel(const half_t*  a, const half_t*  b, float*  c, int K, int lda, int ldb, int ldc)
{
    half_t tA[TT0 * MIIN];
    half_t tB[TT1 * MIIN];
    float  tC[TT0 * TT1 * MFMA_OU * MFMA_OG];

    // pointer for mfma instruction
    VectorStorage<float, 2>::type const* pA = reinterpret_cast<VectorStorage<float, 2>::type const*>(&tA);
    VectorStorage<float, 2>::type const* pB = reinterpret_cast<VectorStorage<float, 2>::type const*>(&tB);
    VectorStorage<float,16>::type const* pC = reinterpret_cast<VectorStorage<float,16>::type const*>(&tC);
    VectorStorage<float,16>::type      * pD = reinterpret_cast<VectorStorage<float,16>::type      *>(&tC);

    int wave_id  = hipThreadIdx_x / WAVE_SIZE;
    int wave_id0 = wave_id % 2;
    int wave_id1 = wave_id / 2;

    int lane_id = hipThreadIdx_x % WAVE_SIZE;
    int tile_id = lane_id % MFMA_T;
    int unro_id = lane_id / MFMA_T;

    // initial tC
    for (int i=0; i < TT0 * TT1 * MFMA_OU * MFMA_OG; i++)
    {
        tC[i] = 0.0f;
    }

    a += hipBlockIdx_x * MT0;
    b += hipBlockIdx_y * MT1;

    // wrap offset
    a += wave_id0 * WT0;
    b += wave_id1 * WT1;

    // thread offset
    a += (tile_id + unro_id * MIIN * lda);
    b += (tile_id + unro_id * MIIN * ldb);

    for (; K>0; K-=DepthU)
    {
        // read a
        for(int i=0; i<TT0; i++) // tile
        {
            for(int j=0; j<MIIN; j++) // unroll
            {
                int dst_index = j       + i * MIIN;
                int src_index = j * lda + i * MFMA_T;
                tA[dst_index] = a[src_index];
            }
        }

        // read b
        for(int i=0; i<TT1; i++) // tile
        {
            for(int j=0; j<MIIN; j++) // unroll
            {
                int dst_index = j       + i * MIIN;
                int src_index = j * ldb + i * MFMA_T;
                tB[dst_index] = b[src_index];
            }
        }

        for(int j=0; j<TT1; j++)
        {
            for(int i=0; i<TT0; i++)
            {
                pD[i + j * TT0] = __builtin_amdgcn_mfma_f32_32x32x8f16(pA[i], pB[j], pC[i + j * TT0], 0, 0, 0);
            }
        }

        a += DepthU * lda;
        b += DepthU * ldb;
    }

    c += (hipBlockIdx_x * MT0     + hipBlockIdx_y * MT1 * ldc);
    c += (wave_id0      * WT0     + wave_id1      * WT1 * ldc);
    c += (unro_id       * MFMA_OU + tile_id             * ldc);

    for (int j=0; j<TT1; j++)
    {
        for (int i=0; i<TT0; i++)
        {
            for (int g=0; g<MFMA_OG; g++)
            {
                for(int u=0; u<MFMA_OU; u++)
                {
                    int src_idx = u + MFMA_OU * (g + MFMA_OG * ( i + TT0 * j));
                    int dst_idx = (u + g * 8 + i * MFMA_T) + (j * MFMA_T * ldc);
                    c[dst_idx]  = tC[src_idx];
                }
            }
        }
    }

}


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
    std::cout << "----- dump " << str << " : -----";
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

  free(hostA);
  free(hostB);
  free(hostC);
  free(ref_C);

  return errors;
}
