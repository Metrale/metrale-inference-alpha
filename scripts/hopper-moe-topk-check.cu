// SPDX-License-Identifier: AGPL-3.0-only
// nvcc -arch=sm_90 -O3 --fmad=false scripts/hopper-moe-topk-check.cu -o /tmp/hopper-moe-topk-check
// Run only in an exclusive GPU window. No model/device allocations outside test.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <random>
#include <limits>
#include "../kernels/gb10/common/moe_topk.cu"
#include "../kernels/hopper/common/moe_topk_warp.cu"
#define OK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s: %s\n",#x,cudaGetErrorString(e));exit(2);}}while(0)
int main(){
 __nv_bfloat16 *d;unsigned *a,*b;float *wa,*wb;
 OK(cudaMalloc(&d,512));OK(cudaMalloc(&a,32));OK(cudaMalloc(&b,32));OK(cudaMalloc(&wa,32));OK(cudaMalloc(&wb,32));
 std::mt19937 rng(42);std::normal_distribution<float> normal(0,7);
 std::vector<__nv_bfloat16> h(256);unsigned ia[8],ib[8];float fa[8],fb[8];
 for(int c=0;c<1036;c++){
  for(int j=0;j<256;j++){
   float x=normal(rng);
   if(c==0)x=1; if(c==1)x=float(j%13);if(c==2)x=float(255-j);
   if(c==3)x=(j%2)?-0.0f:0.0f; if(c==4)x=-3e38f;
   if(c==5)x=-1e30f; if(c==6)x=(j%2)?1e30f:-1e30f;
   if(c>=7&&c<=11 && j==37)x=c==7?std::numeric_limits<float>::quiet_NaN():c==8?INFINITY:c==9?-INFINITY:c==10?0.0f:90.0f;
   if(c>=12)x=float(int(normal(rng)*4))/4;
   h[j]=__float2bfloat16(x);
  }
  OK(cudaMemcpy(d,h.data(),512,cudaMemcpyHostToDevice));
  for(unsigned norm=0;norm<=1;norm++){
   OK(cudaMemset(b,0xcd,32));OK(cudaMemset(wb,0xcd,32));
   moe_topk_softmax<<<1,256>>>(d,a,wa,256,8,norm);
   moe_topk_256_8_warp<<<1,32>>>(d,b,wb,256,8,norm);OK(cudaGetLastError());
   OK(cudaMemcpy(ia,a,32,cudaMemcpyDeviceToHost));OK(cudaMemcpy(ib,b,32,cudaMemcpyDeviceToHost));
   OK(cudaMemcpy(fa,wa,32,cudaMemcpyDeviceToHost));OK(cudaMemcpy(fb,wb,32,cudaMemcpyDeviceToHost));
   if(memcmp(ia,ib,32)||memcmp(fa,fb,32)){
    fprintf(stderr,"FAIL case%d norm%u (strict bitwise including NaNs)\n",c,norm);
    for(int i=0;i<8;i++){unsigned x,y;memcpy(&x,&fa[i],4);memcpy(&y,&fb[i],4);fprintf(stderr,"%u/%u %08x/%08x\n",ia[i],ib[i],x,y);}return 1;
   }
  }
 }
 puts("PASS 2072 bitwise index+FP32 weight cases; random/ties/signedzero/extremes/NaN/infinities; normalize0/1");
 // Graph-based timings match production replay; alternate old/new order.
 for(int round=0;round<4;round++)for(int leg=0;leg<2;leg++){
  bool fast=(leg+round)%2;cudaGraph_t g;cudaGraphExec_t exec;cudaStream_t st;cudaEvent_t t0,t1;
  OK(cudaStreamCreate(&st));OK(cudaEventCreate(&t0));OK(cudaEventCreate(&t1));
  OK(cudaStreamBeginCapture(st,cudaStreamCaptureModeGlobal));
  for(int i=0;i<400;i++){if(fast)moe_topk_256_8_warp<<<1,32,0,st>>>(d,b,wb,256,8,1);else moe_topk_softmax<<<1,256,0,st>>>(d,a,wa,256,8,1);}
  OK(cudaStreamEndCapture(st,&g));OK(cudaGraphInstantiate(&exec,g,nullptr,nullptr,0));
  for(int i=0;i<5;i++)OK(cudaGraphLaunch(exec,st));OK(cudaStreamSynchronize(st));
  OK(cudaEventRecord(t0,st));for(int i=0;i<20;i++)OK(cudaGraphLaunch(exec,st));OK(cudaEventRecord(t1,st));OK(cudaEventSynchronize(t1));
  float ms;OK(cudaEventElapsedTime(&ms,t0,t1));printf("round%d %s graph_us_per_kernel %.6f\n",round,fast?"candidate":"reference",ms*1000/8000);
  OK(cudaGraphExecDestroy(exec));OK(cudaGraphDestroy(g));OK(cudaEventDestroy(t0));OK(cudaEventDestroy(t1));OK(cudaStreamDestroy(st));
 }
 OK(cudaFree(d));OK(cudaFree(a));OK(cudaFree(b));OK(cudaFree(wa));OK(cudaFree(wb));
}
