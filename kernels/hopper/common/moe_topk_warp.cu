// SPDX-License-Identifier: AGPL-3.0-only
// SM90 E256/K8 only; optional single-sequence native-FP8 decode route. One warp holds eight virtual original warps.
// Top-k preserves lower-index tie semantics; softmax preserves the reference
// eight 32-lane sum trees, then serial warp sums/top-k sums in original order.
#include <cuda_bf16.h>
extern "C" __global__ void moe_topk_256_8_warp(
 const __nv_bfloat16* logits, unsigned* indices, float* weights,
 unsigned experts, unsigned topk, unsigned normalize) {
 if (experts != 256 || topk != 8 || blockDim.x != 32) return;
 const unsigned lane=threadIdx.x;
 float v[8], selected[8]; unsigned chosen[8];
 #pragma unroll
 for(int w=0;w<8;w++) v[w]=__bfloat162float(logits[w*32+lane]);
 #pragma unroll
 for(int t=0;t<8;t++) {
  float best=-1e30f; unsigned idx=0;
  #pragma unroll
  for(int w=0;w<8;w++) {
   const unsigned j=w*32+lane;
   if(v[w]>best || (v[w]==best && j<idx)) { best=v[w]; idx=j; }
  }
  #pragma unroll
  for(int d=16;d>0;d>>=1) {
   float b=__shfl_down_sync(0xffffffff,best,d);
   unsigned j=__shfl_down_sync(0xffffffff,idx,d);
   if(b>best || (b==best && j<idx)) {best=b;idx=j;}
  }
  best=__shfl_sync(0xffffffff,best,0); idx=__shfl_sync(0xffffffff,idx,0);
  selected[t]=best; chosen[t]=idx;
  #pragma unroll
  for(int w=0;w<8;w++) if(unsigned(w*32)+lane==idx) v[w]=-1e30f;
 }
 float total=0.0f;
 #pragma unroll
 for(int w=0;w<8;w++) {
  float s=__expf(v[w]-selected[0]);
  #pragma unroll
  for(int d=16;d>0;d>>=1) s+=__shfl_down_sync(0xffffffff,s,d);
  if(lane==0) total+=s;
 }
 if(lane==0) {
  #pragma unroll
  for(int t=0;t<8;t++) total+=__expf(selected[t]-selected[0]);
  float sum=0.0f; const float global_max=selected[0];
  #pragma unroll
  for(int t=0;t<8;t++) {indices[t]=chosen[t];selected[t]=__expf(selected[t]-global_max)/total;}
  if(normalize) {
   #pragma unroll
   for(int t=0;t<8;t++) sum+=selected[t];
  }
  #pragma unroll
  for(int t=0;t<8;t++) weights[t]=normalize ? selected[t]/sum : selected[t];
 }
}
