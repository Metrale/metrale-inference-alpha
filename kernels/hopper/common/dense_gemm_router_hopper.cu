// SPDX-License-Identifier: AGPL-3.0-only
#include <cuda_bf16.h>
// Same BK64 and separate mul/add strict ascending k chain as reference.
template<int BM,int BN> __device__ __forceinline__ void router_exact(
 const __nv_bfloat16* __restrict__ A,const __nv_bfloat16* __restrict__ B,
 __nv_bfloat16* __restrict__ C,unsigned M,unsigned N,unsigned K) {
 constexpr int BK=64,NC=BN/16,NT=BM*16;
 __shared__ float sA[BM][BK+1],sB[BN][BK+1];
 unsigned tid=threadIdx.y*16+threadIdx.x,row0=blockIdx.y*BM,col0=blockIdx.x*BN;
 float acc[NC]={};
 for(unsigned kb=0;kb<K;kb+=BK){
  for(unsigned e=tid*4;e<BM*BK;e+=NT*4){
   unsigned r=e/BK,c=e%BK,gr=row0+r,gc=kb+c;
   if(gr<M && gc+3<K){
    ushort4 v=*(const ushort4*)(A+(size_t)gr*K+gc);
    sA[r][c]=__bfloat162float(__ushort_as_bfloat16(v.x));
    sA[r][c+1]=__bfloat162float(__ushort_as_bfloat16(v.y));
    sA[r][c+2]=__bfloat162float(__ushort_as_bfloat16(v.z));
    sA[r][c+3]=__bfloat162float(__ushort_as_bfloat16(v.w));
   }else for(int j=0;j<4;j++)sA[r][c+j]=(gr<M&&gc+j<K)?__bfloat162float(A[(size_t)gr*K+gc+j]):0.f;
  }
  for(unsigned e=tid*8;e<BN*BK;e+=NT*8){
   unsigned r=e/BK,c=e%BK,gn=col0+r,gc=kb+c;
   if(gn<N&&gc+7<K){
    uint4 v=*(const uint4*)(B+(size_t)gn*K+gc);const unsigned short* u=(const unsigned short*)&v;
    #pragma unroll
    for(int j=0;j<8;j++)sB[r][c+j]=__bfloat162float(__ushort_as_bfloat16(u[j]));
   }else for(int j=0;j<8;j++)sB[r][c+j]=(gn<N&&gc+j<K)?__bfloat162float(B[(size_t)gn*K+gc+j]):0.f;
  }
  __syncthreads();
  #pragma unroll 8
  for(unsigned kk=0;kk<BK;kk++){
   float a=sA[threadIdx.y][kk];
   #pragma unroll
   for(int j=0;j<NC;j++)acc[j]+=a*sB[threadIdx.x*NC+j][kk];
  }
  __syncthreads();
 }
 unsigned row=row0+threadIdx.y,col=col0+threadIdx.x*NC;
 if(row<M){
  #pragma unroll
  for(int j=0;j<NC;j++)if(col+j<N)C[(size_t)row*N+col+j]=__float2bfloat16(acc[j]);
 }
}

// Hopper-only module: preserves --fmad=false and the probed ascending-K body.
extern "C" __global__ void dense_gemm_router_hopper_8x32(
 const __nv_bfloat16* __restrict__ A,const __nv_bfloat16* __restrict__ B,
 __nv_bfloat16* __restrict__ C,unsigned M,unsigned N,unsigned K) {
 router_exact<8,32>(A,B,C,M,N,K);
}
