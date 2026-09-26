// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Raw NCCL bindings used by metrale-comm, plus the CUDA driver
//! stream and event calls it needs. The function declarations follow the
//! installed nccl.h (2.29.7); `NcclConfig` does not (see its doc).
//!
//! The `extern` functions are unchecked: a caller passes a communicator from
//! `ncclCommInitRank`, device pointers valid for the element count on that
//! communicator's device, and a stream of that device. The wrappers below
//! (`create_stream` to `stream_ready`) check only the status code.
//!
//! Owner: metrale-gpu-sys.
//! Invariants: none beyond the types.

use std::ffi::c_void;

/// 2026-09-26: `ncclComm_t`, an opaque communicator handle.
pub type NcclComm = *mut c_void;

/// 2026-09-26: `ncclUniqueId`: 128 bytes, passed by value.
#[repr(C)]
#[derive(Copy, Clone)]
pub struct NcclUniqueId {
    pub internal: [u8; 128],
}

/// 2026-09-26: Meant for `ncclCommInitRankConfig`, which nothing in the
/// workspace calls. It does not match `ncclConfig_t` in the installed nccl.h
/// (2.29.7): that struct starts with magic `NCCL_API_MAGIC` (0xcafebeef), holds
/// `netName` as a pointer, and has more fields.
#[repr(C)]
pub struct NcclConfig {
    pub size: usize,
    pub magic: u32,
    pub version: u32,
    pub blocking: i32,
    pub cga_cluster_size: i32,
    pub min_ctas: i32,
    pub max_ctas: i32,
    pub net_name: [u8; 8],
    pub split_share: i32,
}

impl NcclConfig {
    pub fn non_blocking() -> Self {
        Self {
            size: std::mem::size_of::<Self>(),
            magic: 0x4e43434c,
            version: 22907,
            blocking: 0,
            cga_cluster_size: -1,
            min_ctas: -1,
            max_ctas: -1,
            net_name: [0; 8],
            split_share: -1,
        }
    }
}

/// 2026-09-26: `ncclResult_t`, values 0 to 7 as in nccl.h.
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum NcclResult {
    Success = 0,
    UnhandledCudaError = 1,
    SystemError = 2,
    InternalError = 3,
    InvalidArgument = 4,
    InvalidUsage = 5,
    RemoteError = 6,
    InProgress = 7,
}

/// 2026-09-26: `ncclDataType_t`, the values up to `ncclBfloat16` (9) as in
/// nccl.h.
#[repr(C)]
#[derive(Debug, Copy, Clone)]
#[allow(dead_code)]
pub enum NcclDataType {
    Int8 = 0,
    Uint8 = 1,
    Int32 = 2,
    Uint32 = 3,
    Int64 = 4,
    Uint64 = 5,
    Float16 = 6,
    Float32 = 7,
    Float64 = 8,
    Bfloat16 = 9,
}

/// 2026-09-26: `ncclRedOp_t`'s built-in operations, values as in nccl.h.
#[repr(C)]
#[derive(Debug, Copy, Clone)]
#[allow(dead_code)]
pub enum NcclRedOp {
    Sum = 0,
    Prod = 1,
    Max = 2,
    Min = 3,
    Avg = 4,
}

#[link(name = "nccl")]
unsafe extern "C" {
    pub fn ncclGetUniqueId(id: *mut NcclUniqueId) -> NcclResult;

    pub fn ncclCommInitRank(
        comm: *mut NcclComm,
        nranks: i32,
        id: NcclUniqueId,
        rank: i32,
    ) -> NcclResult;

    pub fn ncclCommInitRankConfig(
        comm: *mut NcclComm,
        nranks: i32,
        id: NcclUniqueId,
        rank: i32,
        config: *const NcclConfig,
    ) -> NcclResult;

    pub fn ncclAllReduce(
        sendbuf: *const c_void,
        recvbuf: *mut c_void,
        count: usize,
        datatype: NcclDataType,
        op: NcclRedOp,
        comm: NcclComm,
        stream: u64,
    ) -> NcclResult;

    pub fn ncclBroadcast(
        sendbuf: *const c_void,
        recvbuf: *mut c_void,
        count: usize,
        datatype: NcclDataType,
        root: i32,
        comm: NcclComm,
        stream: u64,
    ) -> NcclResult;

    pub fn ncclCommDestroy(comm: NcclComm) -> NcclResult;

    pub fn ncclGetErrorString(result: NcclResult) -> *const std::ffi::c_char;

    pub fn ncclCommRegister(
        comm: NcclComm,
        buff: *mut c_void,
        size: usize,
        handle: *mut *mut c_void,
    ) -> NcclResult;

    pub fn ncclCommDeregister(comm: NcclComm, handle: *mut c_void) -> NcclResult;

    pub fn ncclMemAlloc(ptr: *mut *mut c_void, size: usize) -> NcclResult;

    pub fn ncclMemFree(ptr: *mut c_void) -> NcclResult;

    pub fn ncclSend(
        sendbuf: *const c_void,
        count: usize,
        datatype: NcclDataType,
        peer: i32,
        comm: NcclComm,
        stream: u64,
    ) -> NcclResult;

    pub fn ncclRecv(
        recvbuf: *mut c_void,
        count: usize,
        datatype: NcclDataType,
        peer: i32,
        comm: NcclComm,
        stream: u64,
    ) -> NcclResult;

    // 2026-09-26: `recvbuf` holds `nranks * sendcount` elements, rank i's at
    // offset `i * sendcount` (nccl.h).
    pub fn ncclAllGather(
        sendbuf: *const c_void,
        recvbuf: *mut c_void,
        sendcount: usize,
        datatype: NcclDataType,
        comm: NcclComm,
        stream: u64,
    ) -> NcclResult;

    // 2026-09-26: `sendbuf` holds `nranks * recvcount` elements; each rank
    // receives `recvcount` (nccl.h).
    pub fn ncclReduceScatter(
        sendbuf: *const c_void,
        recvbuf: *mut c_void,
        recvcount: usize,
        datatype: NcclDataType,
        op: NcclRedOp,
        comm: NcclComm,
        stream: u64,
    ) -> NcclResult;

    pub fn ncclGroupStart() -> NcclResult;
    pub fn ncclGroupEnd() -> NcclResult;

    pub fn ncclCommGetAsyncError(comm: NcclComm, async_error: *mut NcclResult) -> NcclResult;

    // 2026-09-26: Unlike `ncclCommDestroy`, also aborts operations that may
    // still be running on the device (nccl.h).
    pub fn ncclCommAbort(comm: NcclComm) -> NcclResult;
}

// 2026-09-26: CUDA driver calls for metrale-comm's streams and events.
#[link(name = "cuda")]
unsafe extern "C" {
    fn cuStreamCreate(phStream: *mut u64, flags: u32) -> i32;
    fn cuEventCreate(phEvent: *mut u64, flags: u32) -> i32;
    fn cuEventRecord(hEvent: u64, hStream: u64) -> i32;
    fn cuStreamWaitEvent(hStream: u64, hEvent: u64, flags: u32) -> i32;
    fn cuEventDestroy_v2(hEvent: u64) -> i32;
    fn cuStreamDestroy_v2(hStream: u64) -> i32;
    fn cuStreamSynchronize(hStream: u64) -> i32;
    fn cuStreamQuery(hStream: u64) -> i32;
}

pub fn create_stream() -> anyhow::Result<u64> {
    let mut stream: u64 = 0;
    let status = unsafe { cuStreamCreate(&mut stream, 1) }; // 2026-09-26: CU_STREAM_NON_BLOCKING
    if status != 0 {
        anyhow::bail!("cuStreamCreate failed: status {status}");
    }
    Ok(stream)
}

pub fn create_event() -> anyhow::Result<u64> {
    let mut event: u64 = 0;
    let status = unsafe { cuEventCreate(&mut event, 0x02) }; // 2026-09-26: CU_EVENT_DISABLE_TIMING
    if status != 0 {
        anyhow::bail!("cuEventCreate failed: status {status}");
    }
    Ok(event)
}

pub fn record_event(event: u64, stream: u64) -> anyhow::Result<()> {
    let status = unsafe { cuEventRecord(event, stream) };
    if status != 0 {
        anyhow::bail!("cuEventRecord failed: status {status}");
    }
    Ok(())
}

pub fn stream_wait_event(stream: u64, event: u64) -> anyhow::Result<()> {
    let status = unsafe { cuStreamWaitEvent(stream, event, 0) };
    if status != 0 {
        anyhow::bail!("cuStreamWaitEvent failed: status {status}");
    }
    Ok(())
}

pub fn destroy_event(event: u64) {
    if event != 0 {
        unsafe { cuEventDestroy_v2(event) };
    }
}

pub fn destroy_stream(stream: u64) {
    if stream != 0 {
        unsafe { cuStreamDestroy_v2(stream) };
    }
}

pub fn sync_stream(stream: u64) -> anyhow::Result<()> {
    let status = unsafe { cuStreamSynchronize(stream) };
    if status != 0 {
        anyhow::bail!("cuStreamSynchronize failed: status {status}");
    }
    Ok(())
}

/// 2026-09-26: `ncclMemAlloc`: `size` bytes of device memory from NCCL's
/// allocator; nccl.h notes the allocation may be larger than requested.
///
/// # Safety
/// The returned pointer must be freed with [`nccl_mem_free`].
pub unsafe fn nccl_mem_alloc(size: usize) -> anyhow::Result<*mut c_void> {
    let mut ptr: *mut c_void = std::ptr::null_mut();
    let result = unsafe { ncclMemAlloc(&mut ptr, size) };
    check_nccl(result, "ncclMemAlloc")?;
    Ok(ptr)
}

/// 2026-09-26: `ncclMemFree` for a pointer from [`nccl_mem_alloc`].
///
/// # Safety
/// `ptr` must have been returned by [`nccl_mem_alloc`] and not yet freed.
pub unsafe fn nccl_mem_free(ptr: *mut c_void) -> anyhow::Result<()> {
    let result = unsafe { ncclMemFree(ptr) };
    check_nccl(result, "ncclMemFree")
}

/// 2026-09-26: `Ok` for `Success`; otherwise an error naming `context`, the
/// text of `ncclGetErrorString` and the code.
pub fn check_nccl(result: NcclResult, context: &str) -> anyhow::Result<()> {
    if result == NcclResult::Success {
        Ok(())
    } else {
        let msg = unsafe {
            let ptr = ncclGetErrorString(result);
            if ptr.is_null() {
                format!("{result:?}")
            } else {
                std::ffi::CStr::from_ptr(ptr).to_string_lossy().into_owned()
            }
        };
        anyhow::bail!("NCCL error in {context}: {msg} ({result:?})")
    }
}

/// 2026-09-26: `cuStreamQuery`, without waiting: `true` when the stream's work
/// is done, `false` on `CUDA_ERROR_NOT_READY` (600), an error otherwise.
pub fn stream_ready(stream: u64) -> anyhow::Result<bool> {
    match unsafe { cuStreamQuery(stream) } {
        0 => Ok(true),
        600 => Ok(false),
        status => anyhow::bail!("cuStreamQuery failed: status {status}"),
    }
}
