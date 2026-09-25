// SPDX-License-Identifier: AGPL-3.0-only
// Exactly ordered E<=256 worklist. No floating-point arithmetic or atomics.
#include <cassert>
extern "C" __global__ void moe_build_tile_worklist_ordered(
 const int* offsets,const unsigned long long* weights,unsigned* list,int* total,
 unsigned experts,unsigned nt,unsigned mt) {
 __shared__ unsigned prefix[257];
 const unsigned t=threadIdx.x;
 assert(experts<=256 && nt<=64 && mt>0 && blockDim.x==256);
 unsigned count=0;
 if(t<experts){int rows=offsets[t+1]-offsets[t];if(rows>0 && weights[t]) count=((unsigned(rows)+mt-1)/mt)*nt;}
 prefix[t+1]=count; if(t==0)prefix[0]=0;
 __syncthreads();
 // One scalar scan of 256 SHARED words, instead of thousands of global stores.
 if(t==0){for(unsigned e=1;e<=experts;e++)prefix[e]+=prefix[e-1];*total=int(prefix[experts]);}
 __syncthreads();
 for(unsigned i=t;i<prefix[experts];i+=256){
  unsigned lo=0,hi=experts;
  while(lo<hi){unsigned mid=(lo+hi)/2;if(prefix[mid+1]<=i)lo=mid+1;else hi=mid;}
  unsigned tile=i-prefix[lo],m=tile/nt,n=tile%nt;
  assert(m<(1u<<26));list[i*2]=lo;list[i*2+1]=(m<<6)|n;
 }
}
