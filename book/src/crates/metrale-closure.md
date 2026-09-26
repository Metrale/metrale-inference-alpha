# metrale-closure

**Path:** `crates/closure/`

What a kernel target is actually compiled from, as one hash: the transitive quoted-`#include` closure of the target's resolved sources, plus the inputs that change emitted code without appearing in any source (nvcc flags, `MODEL.toml`, `KERNEL.toml`, `HARDWARE.toml`, the arch string and the compiler version). A resolved file set is not enough, because a shadow file may `#include` the file it shadows and headers are in no set at all. The crate documentation lists what the hash does not cover (angle-bracket includes, include search paths, preprocessor conditionals). The kernel-layout resolver (`src/layout.rs`) also lives here.
