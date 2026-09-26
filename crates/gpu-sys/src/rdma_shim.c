// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-26: One-sided RDMA verbs shim over libibverbs, bound by `verbs.rs`.
// `ibv_post_send` and `ibv_poll_cq` are `static inline` in
// <infiniband/verbs.h>, so Rust cannot bind them; this translation unit wraps
// them, the RC QP lifecycle (create -> INIT -> RTR -> RTS) and MR registration
// behind a small C ABI.
//
// One `rs_conn` is one device context, PD, CQ and RC QP on port 1, plus a
// table of at most RS_MAX_MR registered MRs, deregistered by `rs_destroy`.
//
// Owner: metrale-gpu-sys.
// Invariants:
// - `rs_create` returns NULL on any failure and leaks nothing: it frees the
//   device list on every path and hands a partial `rs_conn` to `rs_destroy`.



#include <infiniband/verbs.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define RS_MAX_MR 512

struct rs_conn {
    struct ibv_context *ctx;
    struct ibv_pd *pd;
    struct ibv_cq *cq;
    struct ibv_qp *qp;
    union ibv_gid gid;
    struct ibv_mr *mrs[RS_MAX_MR];
    int n_mr;
    uint8_t port;
    int gid_idx;
};

void rs_destroy(struct rs_conn *c);

// 2026-09-26: Open device `dev_name` (port 1), read the GID at `gid_idx`,
// allocate PD and CQ, create an RC QP and move it to INIT. NULL on failure.
struct rs_conn *rs_create(const char *dev_name, int gid_idx) {
    struct ibv_device **list = ibv_get_device_list(NULL);
    if (!list) {
        return NULL;
    }
    struct ibv_device *dev = NULL;
    for (int i = 0; list[i]; i++) {
        if (strcmp(ibv_get_device_name(list[i]), dev_name) == 0) {
            dev = list[i];
            break;
        }
    }
    if (!dev) {
        ibv_free_device_list(list);
        return NULL;
    }
    struct rs_conn *c = calloc(1, sizeof(*c));
    if (!c) {
        ibv_free_device_list(list);
        return NULL;
    }
    c->port = 1;
    c->gid_idx = gid_idx;
    c->ctx = ibv_open_device(dev);
    ibv_free_device_list(list);
    if (!c->ctx) {
        goto err;
    }
    if (ibv_query_gid(c->ctx, c->port, gid_idx, &c->gid)) {
        goto err;
    }
    c->pd = ibv_alloc_pd(c->ctx);
    if (!c->pd) {
        goto err;
    }
    c->cq = ibv_create_cq(c->ctx, 256, NULL, NULL, 0);
    if (!c->cq) {
        goto err;
    }
    struct ibv_qp_init_attr qa;
    memset(&qa, 0, sizeof(qa));
    qa.send_cq = c->cq;
    qa.recv_cq = c->cq;
    qa.qp_type = IBV_QPT_RC;
    qa.cap.max_send_wr = 256;
    qa.cap.max_recv_wr = 16;
    qa.cap.max_send_sge = 1;
    qa.cap.max_recv_sge = 1;
    c->qp = ibv_create_qp(c->pd, &qa);
    if (!c->qp) {
        goto err;
    }
    struct ibv_qp_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_INIT;
    attr.pkey_index = 0;
    attr.port_num = c->port;
    // 2026-09-26: Every QP allows remote read and write, because the cache
    // peer's QP receives RDMA WRITEs. Each MR's own flags (`rs_reg_mr`) still
    // apply: the expert and weight peers register their stores for remote read
    // only.
    attr.qp_access_flags =
        IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_LOCAL_WRITE;
    if (ibv_modify_qp(c->qp, &attr,
                      IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT |
                          IBV_QP_ACCESS_FLAGS)) {
        goto err;
    }
    return c;
err:
    rs_destroy(c);
    return NULL;
}

uint32_t rs_qpn(struct rs_conn *c) { return c->qp->qp_num; }

// 2026-09-26: Copy the 16-byte GID read at `gid_idx` into `out`.
void rs_gid(struct rs_conn *c, uint8_t out[16]) { memcpy(out, c->gid.raw, 16); }

// 2026-09-26: Register `[addr, addr+len)` as an MR. `flags`:
//   bit 0 (1) -> REMOTE_READ;
//   bit 1 (2) -> REMOTE_WRITE and LOCAL_WRITE (ibv_reg_mr(3) requires
//                LOCAL_WRITE with REMOTE_WRITE);
//   0         -> LOCAL_WRITE.
// Flag 1 alone leaves out LOCAL_WRITE, so the peers' read-only mappings
// (`Mmap::open_ro`) register without write access. Callers: the peers' stores
// pass 1, client buffers 0, the cache peer's arena 3. Returns -1 when the MR
// table is full or `ibv_reg_mr` fails.

int rs_reg_mr(struct rs_conn *c, void *addr, size_t len, int flags,
              uint32_t *lkey, uint32_t *rkey) {
    if (c->n_mr >= RS_MAX_MR) {
        return -1;
    }
    int access = 0;
    if (flags & 1) {
        access |= IBV_ACCESS_REMOTE_READ;
    }
    if (flags & 2) {
        access |= IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_LOCAL_WRITE;
    }
    if (access == 0) {
        access = IBV_ACCESS_LOCAL_WRITE;
    }
    struct ibv_mr *mr = ibv_reg_mr(c->pd, addr, len, access);
    if (!mr) {
        return -1;
    }
    c->mrs[c->n_mr++] = mr;
    *lkey = mr->lkey;
    *rkey = mr->rkey;
    return 0;
}

// 2026-09-26: Move the QP INIT -> RTR -> RTS towards the remote QP.
// `remote_psn` is the remote's send PSN (our rq_psn); `local_psn` is our send
// PSN (sq_psn). The path MTU is the port's active MTU. Returns 0, -1 when the
// port query fails, -2 for RTR, -3 for RTS.
int rs_connect(struct rs_conn *c, uint32_t remote_qpn, uint32_t remote_psn,
               uint32_t local_psn, const uint8_t remote_gid[16]) {
    struct ibv_port_attr pa;
    if (ibv_query_port(c->ctx, c->port, &pa)) {
        return -1;
    }
    struct ibv_qp_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTR;
    attr.path_mtu = pa.active_mtu;
    attr.dest_qp_num = remote_qpn;
    attr.rq_psn = remote_psn;
    attr.max_dest_rd_atomic = 16;
    attr.min_rnr_timer = 12;
    attr.ah_attr.is_global = 1;
    attr.ah_attr.port_num = c->port;
    attr.ah_attr.grh.hop_limit = 64;
    attr.ah_attr.grh.sgid_index = c->gid_idx;
    attr.ah_attr.grh.traffic_class = 0;
    memcpy(attr.ah_attr.grh.dgid.raw, remote_gid, 16);
    if (ibv_modify_qp(c->qp, &attr,
                      IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU |
                          IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                          IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER)) {
        return -2;
    }
    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTS;
    attr.timeout = 14;
    attr.retry_cnt = 7;
    attr.rnr_retry = 7;
    attr.sq_psn = local_psn;
    attr.max_rd_atomic = 16;
    if (ibv_modify_qp(c->qp, &attr,
                      IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
                          IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN |
                          IBV_QP_MAX_QP_RD_ATOMIC)) {
        return -3;
    }
    return 0;
}

// 2026-09-26: Post one RDMA READ of `len` bytes from `remote_addr` (`rkey`)
// into `local_addr` (`lkey`). Signaled, so it produces a completion tagged
// `wr_id`. Returns `ibv_post_send`'s result.
int rs_post_read(struct rs_conn *c, void *local_addr, uint32_t lkey,
                 uint64_t remote_addr, uint32_t rkey, uint32_t len,
                 uint64_t wr_id) {
    struct ibv_sge sge;
    memset(&sge, 0, sizeof(sge));
    sge.addr = (uintptr_t)local_addr;
    sge.length = len;
    sge.lkey = lkey;
    struct ibv_send_wr wr;
    memset(&wr, 0, sizeof(wr));
    wr.wr_id = wr_id;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_RDMA_READ;
    wr.send_flags = IBV_SEND_SIGNALED;
    wr.wr.rdma.remote_addr = remote_addr;
    wr.wr.rdma.rkey = rkey;
    struct ibv_send_wr *bad = NULL;
    return ibv_post_send(c->qp, &wr, &bad);
}

// 2026-09-26: Post one RDMA WRITE of `len` bytes from `local_addr` (`lkey`)
// to `remote_addr` (`rkey`). Signaled, so it produces a completion tagged
// `wr_id`. Returns `ibv_post_send`'s result.

int rs_post_write(struct rs_conn *c, void *local_addr, uint32_t lkey,
                  uint64_t remote_addr, uint32_t rkey, uint32_t len,
                  uint64_t wr_id) {
    struct ibv_sge sge;
    memset(&sge, 0, sizeof(sge));
    sge.addr = (uintptr_t)local_addr;
    sge.length = len;
    sge.lkey = lkey;
    struct ibv_send_wr wr;
    memset(&wr, 0, sizeof(wr));
    wr.wr_id = wr_id;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_RDMA_WRITE;
    wr.send_flags = IBV_SEND_SIGNALED;
    wr.wr.rdma.remote_addr = remote_addr;
    wr.wr.rdma.rkey = rkey;
    struct ibv_send_wr *bad = NULL;
    return ibv_post_send(c->qp, &wr, &bad);
}

// 2026-09-26: Busy-poll for one completion and write its id to *out_wr_id.
// Returns 0 on success, the positive `ibv_wc_status` of a failed completion
// (the id is still written), or -1 when `ibv_poll_cq` fails.
int rs_poll(struct rs_conn *c, uint64_t *out_wr_id) {
    struct ibv_wc wc;
    for (;;) {
        int n = ibv_poll_cq(c->cq, 1, &wc);
        if (n < 0) {
            return -1;
        }
        if (n == 0) {
            continue;
        }
        *out_wr_id = wc.wr_id;
        if (wc.status != IBV_WC_SUCCESS) {
            return (int)wc.status;
        }
        return 0;
    }
}

void rs_destroy(struct rs_conn *c) {
    if (!c) {
        return;
    }
    if (c->qp) {
        ibv_destroy_qp(c->qp);
    }
    for (int i = 0; i < c->n_mr; i++) {
        if (c->mrs[i]) {
            ibv_dereg_mr(c->mrs[i]);
        }
    }
    if (c->cq) {
        ibv_destroy_cq(c->cq);
    }
    if (c->pd) {
        ibv_dealloc_pd(c->pd);
    }
    if (c->ctx) {
        ibv_close_device(c->ctx);
    }
    free(c);
}
