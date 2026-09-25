// SPDX-License-Identifier: AGPL-3.0-only
#include <cassert>
// E<=256; exact ordered M128 large / M16 small lists, threshold on device.
extern "C" __global__ void bucket_builder(
 const int* off,const unsigned long long* wp,unsigned* large,int* nl,
 unsigned* small,int* ns,unsigned experts,unsigned nt,unsigned min_small_tiles){
 __shared__ unsigned p[2][257];unsigned t=threadIdx.x;
 assert(experts<=256 && nt<=64 && blockDim.x==256);
 unsigned lc=0,sc=0;
 if(t<experts){int n=off[t+1]-off[t];if(wp[t]&&n>0){if(n<=16)sc=nt;else lc=((unsigned(n)+127)/128)*nt;}}
 p[0][t+1]=lc;p[1][t+1]=sc;if(t==0){p[0][0]=0;p[1][0]=0;}
 __syncthreads();
 if(t==0){
  unsigned total_small=0;for(unsigned e=1;e<=experts;e++)total_small+=p[1][e];
  // A sparse one-warp bucket pays its long-K latency without enough CTAs to
  // hide it. Fold every small expert back into M128, preserving list order.
  // Small experts have <=16 rows, so their tile count is nt at BOTH geometries.
  if(total_small<min_small_tiles)for(unsigned e=1;e<=experts;e++){p[0][e]+=p[1][e];p[1][e]=0;}
  for(unsigned e=1;e<=experts;e++){p[0][e]+=p[0][e-1];p[1][e]+=p[1][e-1];}
  *nl=p[0][experts];*ns=p[1][experts];
 }
 __syncthreads();
 #pragma unroll
 for(int bucket=0;bucket<2;bucket++){
  unsigned* list=bucket?small:large;
  for(unsigned i=t;i<p[bucket][experts];i+=256){
   unsigned lo=0,hi=experts;
   while(lo<hi){unsigned mid=(lo+hi)/2;if(p[bucket][mid+1]<=i)lo=mid+1;else hi=mid;}
   unsigned tile=i-p[bucket][lo],m=tile/nt,n=tile%nt;assert(m<(1u<<26));
   list[i*2]=lo;list[i*2+1]=(m<<6)|n;
  }
 }
}
