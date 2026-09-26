// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Rust wrapper over the verbs C shim (`rdma_shim.c`), compiled only
//! under `cfg(metrale_rdma_verbs)`. One `Verbs` is one RC QP with its own device
//! context, PD and CQ. The caller exchanges QP identities (qpn, psn, gid) over
//! TCP; `connect` moves the QP INIT -> RTR -> RTS; `post_read` and `post_write`
//! post one-sided operations and `poll` busy-waits for one completion.
//!
//! Owner: metrale-gpu-sys.
//! Invariants:
//! - `conn` is non-null from `create` until `Drop` destroys it.

use anyhow::{Result, bail};
use std::ffi::CString;
use std::os::raw::{c_char, c_int, c_void};

/// 2026-09-26: A 16-byte GID, sent verbatim over the control channel.
pub type Gid = [u8; 16];

#[repr(C)]
struct RsConn {
    _private: [u8; 0],
}

unsafe extern "C" {
    fn rs_create(dev_name: *const c_char, gid_idx: c_int) -> *mut RsConn;
    fn rs_destroy(c: *mut RsConn);
    fn rs_qpn(c: *mut RsConn) -> u32;
    fn rs_gid(c: *mut RsConn, out: *mut u8);
    fn rs_reg_mr(
        c: *mut RsConn,
        addr: *mut c_void,
        len: usize,
        flags: c_int,
        lkey: *mut u32,
        rkey: *mut u32,
    ) -> c_int;
    fn rs_connect(
        c: *mut RsConn,
        remote_qpn: u32,
        remote_psn: u32,
        local_psn: u32,
        remote_gid: *const u8,
    ) -> c_int;
    fn rs_post_read(
        c: *mut RsConn,
        local_addr: *mut c_void,
        lkey: u32,
        remote_addr: u64,
        rkey: u32,
        len: u32,
        wr_id: u64,
    ) -> c_int;
    fn rs_post_write(
        c: *mut RsConn,
        local_addr: *mut c_void,
        lkey: u32,
        remote_addr: u64,
        rkey: u32,
        len: u32,
        wr_id: u64,
    ) -> c_int;
    fn rs_poll(c: *mut RsConn, out_wr_id: *mut u64) -> c_int;
}

/// 2026-09-26: The keys of a registered memory region.
#[derive(Clone, Copy, Debug)]
pub struct MrKeys {
    pub lkey: u32,
    pub rkey: u32,
}

/// 2026-09-26: One RC QP. Owns the C connection and destroys it on drop.
pub struct Verbs {
    conn: *mut RsConn,
    /// 2026-09-26: This QP's send PSN; the peer uses it as its receive PSN.
    local_psn: u32,
}

// 2026-09-26: SAFETY: `Verbs` is `Send` but not `Sync`, so one thread at a time
// uses the connection. The storage tiers that own a `Verbs` need `Send`, e.g.
// `ExpertTier: Send`.
unsafe impl Send for Verbs {}

impl Verbs {
    /// 2026-09-26: Open device `dev_name` on port 1, read the GID at `gid_idx`
    /// and create an RC QP in INIT. `local_psn` is the QP's send PSN, used by
    /// `connect`.
    ///
    /// # Errors
    /// When the shim returns null: no such device, or a verbs call failed.
    pub fn create(dev_name: &str, gid_idx: u32, local_psn: u32) -> Result<Self> {
        let dev = CString::new(dev_name).unwrap_or_default();
        // 2026-09-26: SAFETY: `dev` is a NUL-terminated string that outlives
        // the call.
        let conn = unsafe { rs_create(dev.as_ptr(), gid_idx as c_int) };
        if conn.is_null() {
            bail!(
                "rs_create failed for device '{dev_name}' gid_idx {gid_idx} \
                 (check `ibv_devinfo -d {dev_name}` link state / GID)"
            );
        }
        Ok(Self { conn, local_psn })
    }

    pub fn qpn(&self) -> u32 {
        // 2026-09-26: SAFETY: `conn` is live for `self`'s lifetime.
        unsafe { rs_qpn(self.conn) }
    }

    pub fn psn(&self) -> u32 {
        self.local_psn
    }

    pub fn gid(&self) -> Gid {
        let mut g = [0u8; 16];
        // 2026-09-26: SAFETY: `conn` is live; `rs_gid` copies 16 bytes into
        // `g`, which holds 16.
        unsafe { rs_gid(self.conn, g.as_mut_ptr()) };
        g
    }

    /// 2026-09-26: Register `[addr, addr+len)`: remote read only when
    /// `remote_read` (the peers' read-only stores), otherwise local write only
    /// (the clients' landing buffers).
    ///
    /// # Errors
    /// When the shim's table of `RS_MAX_MR` regions is full or `ibv_reg_mr`
    /// fails.
    ///
    /// # Safety
    /// `addr` must point at `len` bytes that outlive this `Verbs`: the region
    /// is deregistered only on drop, and the NIC may access it until then.
    pub unsafe fn reg_mr(
        &mut self,
        addr: *mut c_void,
        len: usize,
        remote_read: bool,
    ) -> Result<MrKeys> {
        let mut lkey = 0u32;
        let mut rkey = 0u32;
        // 2026-09-26: SAFETY: `conn` is live, `lkey` and `rkey` are valid
        // out-pointers, and the caller upholds `addr`/`len`.
        let rc = unsafe {
            rs_reg_mr(
                self.conn,
                addr,
                len,
                remote_read as c_int,
                &mut lkey,
                &mut rkey,
            )
        };
        if rc != 0 {
            bail!("ibv_reg_mr failed (addr {addr:p} len {len} remote_read {remote_read})");
        }
        Ok(MrKeys { lkey, rkey })
    }

    /// 2026-09-26: Register `[addr, addr+len)` for remote read, remote write and
    /// local write. The cache peer registers its arena this way.
    ///
    /// # Safety
    /// As for [`Verbs::reg_mr`]: `addr` must back `len` bytes that outlive
    /// `self`.
    pub unsafe fn reg_mr_rw(&mut self, addr: *mut c_void, len: usize) -> Result<MrKeys> {
        let mut lkey = 0u32;
        let mut rkey = 0u32;
        // 2026-09-26: `rs_reg_mr` maps flags 3 to REMOTE_READ | REMOTE_WRITE |
        // LOCAL_WRITE. SAFETY: `conn` is live, the out-pointers are valid, and
        // the caller upholds `addr`/`len`.
        let rc = unsafe { rs_reg_mr(self.conn, addr, len, 3, &mut lkey, &mut rkey) };
        if rc != 0 {
            bail!("ibv_reg_mr(RW) failed (addr {addr:p} len {len})");
        }
        Ok(MrKeys { lkey, rkey })
    }

    /// 2026-09-26: Move the QP INIT -> RTR -> RTS towards the remote QP.
    pub fn connect(&mut self, remote_qpn: u32, remote_psn: u32, remote_gid: &Gid) -> Result<()> {
        // 2026-09-26: SAFETY: `conn` is live; `remote_gid` is 16 bytes.
        let rc = unsafe {
            rs_connect(
                self.conn,
                remote_qpn,
                remote_psn,
                self.local_psn,
                remote_gid.as_ptr(),
            )
        };
        match rc {
            0 => Ok(()),
            -1 => bail!("rs_connect: ibv_query_port failed"),
            -2 => bail!("rs_connect: modify_qp -> RTR failed (check MTU/GID/dest_qpn)"),
            -3 => bail!("rs_connect: modify_qp -> RTS failed"),
            other => bail!("rs_connect: unexpected code {other}"),
        }
    }

    /// 2026-09-26: Post a signaled one-sided READ of `len` bytes from
    /// `remote_addr` (`rkey`) into `local_addr` (`lkey`), tagged `wr_id`. It
    /// does not wait; `poll` reaps the completion.
    ///
    /// # Safety
    /// `local_addr..+len` must lie inside a live region registered under
    /// `lkey`. The buffer and its region must stay live, and untouched by the
    /// CPU, until `poll` returns this `wr_id`: the NIC writes into it until then.
    pub unsafe fn post_read(
        &mut self,
        local_addr: *mut c_void,
        lkey: u32,
        remote_addr: u64,
        rkey: u32,
        len: u32,
        wr_id: u64,
    ) -> Result<()> {
        // 2026-09-26: SAFETY: `conn` is live; the caller upholds
        // `local_addr`/`lkey`.
        let rc =
            unsafe { rs_post_read(self.conn, local_addr, lkey, remote_addr, rkey, len, wr_id) };
        if rc != 0 {
            bail!("ibv_post_send(RDMA_READ) failed: {rc}");
        }
        Ok(())
    }

    /// 2026-09-26: Post a signaled one-sided WRITE of `len` bytes from
    /// `local_addr` (`lkey`) to `remote_addr` (`rkey`), tagged `wr_id`. It does
    /// not wait; `poll` reaps the completion.
    ///
    /// # Safety
    /// `local_addr..+len` must lie inside a live region registered under
    /// `lkey`. The buffer and its region must stay live and unmodified until
    /// `poll` returns this `wr_id`: the NIC reads from it until then.
    pub unsafe fn post_write(
        &mut self,
        local_addr: *mut c_void,
        lkey: u32,
        remote_addr: u64,
        rkey: u32,
        len: u32,
        wr_id: u64,
    ) -> Result<()> {
        // 2026-09-26: SAFETY: `conn` is live; the caller upholds
        // `local_addr`/`lkey`.
        let rc =
            unsafe { rs_post_write(self.conn, local_addr, lkey, remote_addr, rkey, len, wr_id) };
        if rc != 0 {
            bail!("ibv_post_send(RDMA_WRITE) failed: {rc}");
        }
        Ok(())
    }

    /// 2026-09-26: Busy-wait for one completion and return its `wr_id`.
    ///
    /// # Errors
    /// On a completion whose `ibv_wc_status` is not success, or when
    /// `ibv_poll_cq` fails.
    pub fn poll(&mut self) -> Result<u64> {
        let mut wr_id = 0u64;
        // 2026-09-26: SAFETY: `conn` is live; `wr_id` is a valid out-pointer.
        let rc = unsafe { rs_poll(self.conn, &mut wr_id) };
        if rc == 0 {
            Ok(wr_id)
        } else if rc < 0 {
            bail!("ibv_poll_cq error");
        } else {
            bail!("RDMA completion error: ibv_wc_status {rc}");
        }
    }
}

impl Drop for Verbs {
    fn drop(&mut self) {
        // 2026-09-26: SAFETY: `conn` came from `rs_create` and is destroyed
        // only here.
        unsafe { rs_destroy(self.conn) };
    }
}
