// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: CUDA driver API shim for the HIP target: each `cu*` entry point
// below forwards to its HIP equivalent. Built as `libcuda.so` in OUT_DIR, which
// build.rs adds to the link search path (Linux), or into the HIP `cuda.dll`
// (Windows), so the runtime's CUDA driver calls run on HIP.
//
// Owner: HIP shims (kernels crate).
// Invariants: device pointers (`CUdeviceptr`, a u64) and opaque handles are
// cast to their HIP types. Forwarding entry points return HIP's error code;
// cuGetErrorName and cuGetErrorString describe it with the HIP functions.

#include <hip/hip_runtime.h>
#include <cstring>

typedef unsigned long long CUdeviceptr;
extern "C" {


int cuCtxGetCurrent(void** pctx)            { return hipCtxGetCurrent((hipCtx_t*)pctx); }
int cuCtxSetCurrent(void* ctx)              { return hipCtxSetCurrent((hipCtx_t)ctx); }
int cuCtxCreate_v2(void** pctx, unsigned f, int dev)
                                            { return hipCtxCreate((hipCtx_t*)pctx, f, dev); }
int cuCtxDestroy_v2(void* ctx)              { return hipCtxDestroy((hipCtx_t)ctx); }


int cuGetErrorName(int err, const char** s)   { *s = hipGetErrorName((hipError_t)err);   return 0; }
int cuGetErrorString(int err, const char** s) { *s = hipGetErrorString((hipError_t)err); return 0; }


int cuMemAlloc_v2(CUdeviceptr* dptr, size_t n)      { return hipMalloc((void**)dptr, n); }
int cuMemFree_v2(CUdeviceptr dptr)                  { return hipFree((void*)dptr); }
int cuMemAllocHost_v2(void** pp, size_t n)          { return hipHostMalloc(pp, n, 0); }
int cuMemFreeHost(void* p)                          { return hipHostFree(p); }
int cuMemAllocManaged(CUdeviceptr* dptr, size_t n, unsigned flags)
                                                    { return hipMallocManaged((void**)dptr, n, flags); }
int cuMemGetInfo_v2(size_t* free, size_t* total)    { return hipMemGetInfo(free, total); }

int cuMemcpyHtoDAsync_v2(CUdeviceptr dst, const void* src, size_t n, void* s)
                              { return hipMemcpyHtoDAsync((hipDeviceptr_t)dst, (void*)src, n, (hipStream_t)s); }
int cuMemcpyDtoHAsync_v2(void* dst, CUdeviceptr src, size_t n, void* s)
                              { return hipMemcpyDtoHAsync(dst, (hipDeviceptr_t)src, n, (hipStream_t)s); }
int cuMemcpyDtoDAsync_v2(CUdeviceptr dst, CUdeviceptr src, size_t n, void* s)
                              { return hipMemcpyDtoDAsync((hipDeviceptr_t)dst, (hipDeviceptr_t)src, n, (hipStream_t)s); }
int cuMemsetD8Async(CUdeviceptr dst, unsigned char uc, size_t n, void* s)
                              { return hipMemsetD8Async((hipDeviceptr_t)dst, uc, n, (hipStream_t)s); }
int cuMemsetD32Async(CUdeviceptr dst, unsigned int ui, size_t n, void* s)
                              { return hipMemsetD32Async((hipDeviceptr_t)dst, ui, n, (hipStream_t)s); }


int cuModuleLoadData(void** m, const void* image)          { return hipModuleLoadData((hipModule_t*)m, image); }
int cuModuleGetFunction(void** f, void* m, const char* nm) { return hipModuleGetFunction((hipFunction_t*)f, (hipModule_t)m, nm); }
// 2026-09-25: Device address and size of a `__constant__` or global symbol (`registry::device_symbol`).
int cuModuleGetGlobal_v2(CUdeviceptr* dptr, size_t* bytes, void* m, const char* nm)
                              { return hipModuleGetGlobal((hipDeviceptr_t*)dptr, bytes, (hipModule_t)m, nm); }
int cuModuleUnload(void* m)                                { return hipModuleUnload((hipModule_t)m); }
// 2026-09-25: Ignores the attribute and returns success. A launch passes its
// dynamic shared-memory size to `hipModuleLaunchKernel` (`cuLaunchKernel`
// below).
int cuFuncSetAttribute(void* f, int attr, int val)         { (void)f;(void)attr;(void)val; return 0; }
int cuLaunchKernel(void* f, unsigned gx, unsigned gy, unsigned gz,
                   unsigned bx, unsigned by, unsigned bz,
                   unsigned shmem, void* stream, void** params, void** extra) {
  return hipModuleLaunchKernel((hipFunction_t)f, gx, gy, gz, bx, by, bz,
                               shmem, (hipStream_t)stream, params, extra);
}


int cuStreamCreate(void** s, unsigned flags)        { return hipStreamCreateWithFlags((hipStream_t*)s, flags); }
int cuStreamSynchronize(void* s)                    { return hipStreamSynchronize((hipStream_t)s); }
// 2026-09-25: Non-blocking completion poll. Its caller in metrale-gpu-runtime
// (`gpu_copy.rs`, `METRALE_D2H_SPIN_SYNC`) keeps spinning only while the status
// is 600, CUDA's `CUDA_ERROR_NOT_READY`.
int cuStreamQuery(void* s)                          { return hipStreamQuery((hipStream_t)s); }
int cuStreamWaitEvent(void* s, void* e, unsigned f) { return hipStreamWaitEvent((hipStream_t)s, (hipEvent_t)e, f); }
int cuStreamBeginCapture(void* s, int mode)         { return hipStreamBeginCapture((hipStream_t)s, (hipStreamCaptureMode)mode); }
int cuStreamEndCapture(void* s, void** pgraph)      { return hipStreamEndCapture((hipStream_t)s, (hipGraph_t*)pgraph); }


int cuEventCreate(void** e, unsigned flags) { return hipEventCreateWithFlags((hipEvent_t*)e, flags); }
int cuEventDestroy_v2(void* e)              { return hipEventDestroy((hipEvent_t)e); }
int cuEventRecord(void* e, void* s)         { return hipEventRecord((hipEvent_t)e, (hipStream_t)s); }
int cuEventSynchronize(void* e)             { return hipEventSynchronize((hipEvent_t)e); }
int cuEventElapsedTime(float* ms, void* a, void* b) { return hipEventElapsedTime(ms, (hipEvent_t)a, (hipEvent_t)b); }

// 2026-09-25: Five-parameter form; errNode, logBuf and bufSize are ignored. The
// in-tree caller (metrale-gpu-runtime `cuda_backend.rs`, `metrale_scale`) passes three.
int cuGraphInstantiate(void** pexec, void* graph, void** errNode, char* logBuf, size_t bufSize) {
  (void)errNode; (void)logBuf; (void)bufSize;
  return hipGraphInstantiate((hipGraphExec_t*)pexec, (hipGraph_t)graph, nullptr, nullptr, 0);
}
int cuGraphLaunch(void* exec, void* s)  { return hipGraphLaunch((hipGraphExec_t)exec, (hipStream_t)s); }
int cuGraphExecDestroy(void* exec)      { return hipGraphExecDestroy((hipGraphExec_t)exec); }
int cuGraphDestroy(void* graph)         { return hipGraphDestroy((hipGraph_t)graph); }

int cuGraphInstantiateWithFlags(void** pexec, void* graph, unsigned long long flags){ return hipGraphInstantiateWithFlags((hipGraphExec_t*)pexec,(hipGraph_t)graph,flags); }

int cuMemcpyHtoD_v2(unsigned long long d,const void*s,size_t n){return hipMemcpyHtoD((hipDeviceptr_t)d,(void*)s,n);}
int cuMemcpyDtoH_v2(void*d,unsigned long long s,size_t n){return hipMemcpyDtoH(d,(hipDeviceptr_t)s,n);}
int cuMemcpyDtoD_v2(unsigned long long d,unsigned long long s,size_t n){return hipMemcpyDtoD((hipDeviceptr_t)d,(hipDeviceptr_t)s,n);}
int cuMemsetD8_v2(unsigned long long d,unsigned char v,size_t n){return hipMemsetD8((hipDeviceptr_t)d,v,n);}
int cuMemsetD32_v2(unsigned long long d,unsigned int v,size_t n){return hipMemsetD32((hipDeviceptr_t)d,v,n);}
int cuMemHostAlloc(void**p,size_t n,unsigned int f){return hipHostMalloc(p,n,f);}
int cuMemHostGetDevicePointer_v2(CUdeviceptr* pdptr, void* p, unsigned int f)
                                            { return hipHostGetDevicePointer((void**)pdptr, p, f); }

// 2026-09-25: cuDeviceGetAttribute answers fixed values by CUDA attribute number; cuDeviceGetName reports "AMD-gfx1151".
int cuInit(unsigned int f){return hipInit(f);}
int cuDriverGetVersion(int*v){return hipDriverGetVersion(v);}
int cuDeviceGet(int*d,int o){return hipDeviceGet(d,o);}
int cuDeviceGetCount(int*c){return hipGetDeviceCount(c);}
int cuDeviceTotalMem_v2(size_t*b,int d){return hipDeviceTotalMem(b,d);}
int cuDevicePrimaryCtxRetain(void**c,int d){return hipDevicePrimaryCtxRetain((hipCtx_t*)c,d);}
int cuDevicePrimaryCtxRelease_v2(int d){return hipDevicePrimaryCtxRelease(d);}
int cuCtxSynchronize(void){return hipDeviceSynchronize();}
int cuCtxGetDevice(int*d){return hipGetDevice(d);}
int cuDeviceGetName(char*n,int len,int d){ if(len>0){const char*s="AMD-gfx1151"; int i=0; for(;i<len-1 && s[i];i++) n[i]=s[i]; n[i]=0;} return 0;}
int cuDeviceGetAttribute(int*v,int attr,int dev){ (void)dev; switch(attr){ case 75:*v=12;break; case 76:*v=1;break; case 16:*v=40;break; case 1:*v=1024;break; case 10:*v=32;break; case 8:*v=65536;break; case 18:*v=1;break; case 19:*v=1;break; case 41:*v=1;break; case 36:*v=1500;break; default:*v=0;break;} return 0;}

int cuStreamDestroy_v2(void*s){return hipStreamDestroy((hipStream_t)s);}
int cuStreamDestroy(void*s){return hipStreamDestroy((hipStream_t)s);}

}
