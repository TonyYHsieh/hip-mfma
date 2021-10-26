#pragma once

#include "hip/hip_runtime.h"
#include "half.h"

#define HIP_ASSERT(x) (assert((x)==hipSuccess))

#define TOTAL_THREAD 256
#define WAVE_SIZE 64

// Macro tile
#define MT0 256
#define MT1 128

// wrap tile
#define WT0 128
#define WT1 64

// thread tile
#define TT0 4
#define TT1 2

// unroll
#define DepthU 8

// global read vector
#define GRVW 1

// MFMA instruction
#define MFMA_T 32
#define MFMA_K  8

// input per mi instruction
#define MIIN 4

// output per mi instruction
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
    __shared__ char lA[MT0*DepthU*sizeof(half_t)];
    __shared__ char lB[MT1*DepthU*sizeof(half_t)];

    ////////////////////////////////////////////////////////////////////
    // global init
    ////////////////////////////////////////////////////////////////////

    // global read a
    constexpr int grv_AM = MT0 / GRVW;  // 256
    constexpr int grv_AK = TOTAL_THREAD / grv_AM; // 1
    int tid_a_c = hipThreadIdx_x % grv_AM; // tid in continous side
    int tid_a_s = hipThreadIdx_x / grv_AM; // tid in stride size

    a += ((hipBlockIdx_x * MT0) + (tid_a_c * GRVW + tid_a_s * lda));
    half_t* lA_wptr = reinterpret_cast<half_t*>(lA) + tid_a_c * GRVW + tid_a_s * MT0;

    // global read b
    constexpr int grv_BM = MT1 / GRVW;  // 128
    constexpr int grv_BK = TOTAL_THREAD / grv_BM; // 2
    int tid_b_c = hipThreadIdx_x % grv_BM; // tid in continous side
    int tid_b_s = hipThreadIdx_x / grv_BM; // tid in stride size

    b += ((hipBlockIdx_y * MT1) + (tid_b_c * GRVW + tid_b_s * ldb));
    half_t* lB_wptr = reinterpret_cast<half_t*>(lB) + tid_b_c * GRVW + tid_b_s * MT1;


    ////////////////////////////////////////////////////////////////////
    // local read init: base + wrap offset + thread offset
    ////////////////////////////////////////////////////////////////////

    int wave_id  = hipThreadIdx_x / WAVE_SIZE;
    int wave_id0 = wave_id % 2;
    int wave_id1 = wave_id / 2;

    int lane_id = hipThreadIdx_x % WAVE_SIZE;
    int tile_id = lane_id % MFMA_T;
    int unro_id = lane_id / MFMA_T;

    half_t* lA_rptr = reinterpret_cast<half_t*>(lA) + ((wave_id0 * WT0) + (tile_id + unro_id * MIIN * MT0));
    half_t* lB_rptr = reinterpret_cast<half_t*>(lB) + ((wave_id1 * WT1) + (tile_id + unro_id * MIIN * MT1));


    ////////////////////////////////////////////////////////////////////
    // reg read init
    ////////////////////////////////////////////////////////////////////

    half_t tA[TT0 * MIIN];
    half_t tB[TT1 * MIIN];
    float  tC[TT0 * TT1 * MFMA_OU * MFMA_OG];

    // pointer for mfma instruction
    VectorStorage<float, 2>::type const* pA = reinterpret_cast<VectorStorage<float, 2>::type const*>(&tA);
    VectorStorage<float, 2>::type const* pB = reinterpret_cast<VectorStorage<float, 2>::type const*>(&tB);
    VectorStorage<float,16>::type const* pC = reinterpret_cast<VectorStorage<float,16>::type const*>(&tC);
    VectorStorage<float,16>::type      * pD = reinterpret_cast<VectorStorage<float,16>::type      *>(&tC);

    // initial tC
    for (int i=0; i < TT0 * TT1 * MFMA_OU * MFMA_OG; i++)
    {
        tC[i] = 0.0f;
    }


    ////////////////////////////////////////////////////////////////////
    // main loop
    ////////////////////////////////////////////////////////////////////

    for (; K>0; K-=DepthU)
    {
        // global read a
        for (int i=0; i < DepthU/grv_AK; i++)
        {
            lA_wptr[i * grv_AK * MT0] = a[i * grv_AK * lda];
        }

        // global read b
        for (int i=0; i < DepthU/grv_BK; i++)
        {
            lB_wptr[i * grv_BK * MT1] = b[i * grv_BK * ldb];
        }

        __syncthreads();

        // read a
        for(int i=0; i<TT0; i++) // tile
        {
            for(int j=0; j<MIIN; j++) // unroll
            {
                int dst_index = j       + i * MIIN;
                int src_index = j * MT0 + i * MFMA_T;
                tA[dst_index] = lA_rptr[src_index];
            }
        }

        // read b
        for(int i=0; i<TT1; i++) // tile
        {
            for(int j=0; j<MIIN; j++) // unroll
            {
                int dst_index = j       + i * MIIN;
                int src_index = j * MT1 + i * MFMA_T;
                tB[dst_index] = lB_rptr[src_index];
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


