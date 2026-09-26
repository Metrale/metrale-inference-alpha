// SPDX-License-Identifier: MIT OR Apache-2.0

fn main() {
    println!("cargo:rustc-check-cfg=cfg(metrale_rdma_verbs)");
    println!("cargo:rerun-if-env-changed=DEP_METRALE_RDMA_SHIM_HAS_VERBS");
    if std::env::var_os("DEP_METRALE_RDMA_SHIM_HAS_VERBS").is_some() {
        println!("cargo:rustc-cfg=metrale_rdma_verbs");
    }
}
