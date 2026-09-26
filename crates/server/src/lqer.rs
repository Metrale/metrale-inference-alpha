// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Low-rank corrections for the quantization error of a linear
//! weight: a BF16 `left` [rows × rank] and `right` [rank × cols] pair per
//! layer, their `.lqer` file format, and a directory loader. No code here
//! computes the decomposition below.
//!
//! Owner: server.
//! Invariants: every correction that `parse_lqer_bytes` returns or
//! `write_lqer_bytes` serialises passes `validate`.
//!
//! ```text
//! E = W_bf16 - dequant(Q(W))
//! E ≈ U_k Σ_k V_k^T   (rank-k truncated SVD)
//! ```
//!
//! Nothing outside this module uses it: no correction is applied at
//! inference, and only the tests write `.lqer` files.

use std::path::Path;

/// 2026-09-26: One layer's correction: two BF16 matrices, `left`
/// [rows × rank] and `right` [rank × cols], whose product is rows × cols.
#[derive(Debug, Clone)]
pub struct LqerCorrection {
    pub layer_name: String,
    pub rank: usize,
    pub rows: usize,
    pub cols: usize,
    /// 2026-09-26: BF16 bytes of `left` [rows × rank].
    pub left_bf16: Vec<u8>,
    /// 2026-09-26: BF16 bytes of `right` [rank × cols].
    pub right_bf16: Vec<u8>,
}

impl LqerCorrection {
    /// 2026-09-26: Bytes of the two matrices.
    pub fn memory_bytes(&self) -> usize {
        self.left_bf16.len() + self.right_bf16.len()
    }

    /// 2026-09-26: `Err(reason)` when a matrix's byte length is not its shape
    /// × 2.
    pub fn validate(&self) -> Result<(), String> {
        let expected_left = self.rows * self.rank * 2;
        if self.left_bf16.len() != expected_left {
            return Err(format!(
                "left matrix size mismatch: expected {expected_left}B, got {}B",
                self.left_bf16.len()
            ));
        }
        let expected_right = self.rank * self.cols * 2;
        if self.right_bf16.len() != expected_right {
            return Err(format!(
                "right matrix size mismatch: expected {expected_right}B, got {}B",
                self.right_bf16.len()
            ));
        }
        Ok(())
    }
}

/// 2026-09-26: Corrections keyed by layer name; inserting a name again
/// replaces the earlier correction.
#[derive(Debug, Clone, Default)]
pub struct LqerCorrectionSet {
    by_layer: std::collections::HashMap<String, LqerCorrection>,
}

impl LqerCorrectionSet {
    pub fn empty() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, c: LqerCorrection) {
        self.by_layer.insert(c.layer_name.clone(), c);
    }

    pub fn get(&self, layer: &str) -> Option<&LqerCorrection> {
        self.by_layer.get(layer)
    }

    pub fn len(&self) -> usize {
        self.by_layer.len()
    }

    pub fn is_empty(&self) -> bool {
        self.by_layer.is_empty()
    }

    pub fn total_memory_bytes(&self) -> usize {
        self.by_layer.values().map(|c| c.memory_bytes()).sum()
    }
}

/// 2026-09-26: `ceil(fraction × min(rows, cols))`, at most `min(rows, cols)`
/// and, when both are non-zero, at least 1.
pub fn suggest_rank(rows: usize, cols: usize, fraction: f32) -> usize {
    let bound = rows.min(cols);
    let r = ((bound as f32) * fraction).ceil() as usize;
    r.max(1).min(bound)
}

/// 2026-09-26: The `.lqer` file format, which `parse_lqer_bytes` and
/// `write_lqer_bytes` implement.
///
/// ```text
/// offset  size  field
/// ────────────────────────────────────
/// 0       8     magic = b"METRLLQE"
/// 8       4     format version (u32 little-endian) — currently 1
/// 12      4     rank          (u32 LE)
/// 16      4     rows          (u32 LE)
/// 20      4     cols          (u32 LE)
/// 24      4     name_len      (u32 LE)
/// 28      N     layer_name    (UTF-8, length = name_len)
/// 28+N    PAD   zero-padding to 8-byte alignment
/// ────────────────────────────────────
/// next    rows × rank × 2     left  BF16 matrix (column-major)
/// next    rank × cols × 2     right BF16 matrix (row-major)
/// ```
///
/// The matrix bytes are stored as they are, with no compression.
const LQER_MAGIC: &[u8; 8] = b"METRLLQE";
const LQER_VERSION: u32 = 1;

/// 2026-09-26: Why a `.lqer` file could not be read or parsed.
#[derive(Debug)]
pub enum LqerLoadError {
    Io(std::io::Error),
    BadMagic,
    UnsupportedVersion(u32),
    Truncated { expected: usize, got: usize },
    InvalidName(std::string::FromUtf8Error),
    ShapeMismatch(String),
}

impl std::fmt::Display for LqerLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "io: {e}"),
            Self::BadMagic => write!(f, "bad magic — expected METRLLQE"),
            Self::UnsupportedVersion(v) => write!(f, "unsupported version: {v}"),
            Self::Truncated { expected, got } => {
                write!(f, "truncated: expected {expected} bytes, got {got}")
            }
            Self::InvalidName(e) => write!(f, "invalid utf-8 in layer name: {e}"),
            Self::ShapeMismatch(s) => write!(f, "shape mismatch: {s}"),
        }
    }
}

impl std::error::Error for LqerLoadError {}

impl From<std::io::Error> for LqerLoadError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

/// 2026-09-26: Parse one `.lqer` file's bytes, with no I/O. Bytes after the
/// `right` matrix are ignored.
pub fn parse_lqer_bytes(buf: &[u8]) -> Result<LqerCorrection, LqerLoadError> {
    if buf.len() < 28 {
        return Err(LqerLoadError::Truncated {
            expected: 28,
            got: buf.len(),
        });
    }
    if &buf[0..8] != LQER_MAGIC {
        return Err(LqerLoadError::BadMagic);
    }
    let version = u32::from_le_bytes(buf[8..12].try_into().unwrap());
    if version != LQER_VERSION {
        return Err(LqerLoadError::UnsupportedVersion(version));
    }
    let rank = u32::from_le_bytes(buf[12..16].try_into().unwrap()) as usize;
    let rows = u32::from_le_bytes(buf[16..20].try_into().unwrap()) as usize;
    let cols = u32::from_le_bytes(buf[20..24].try_into().unwrap()) as usize;
    let name_len = u32::from_le_bytes(buf[24..28].try_into().unwrap()) as usize;

    let name_end = 28 + name_len;
    if buf.len() < name_end {
        return Err(LqerLoadError::Truncated {
            expected: name_end,
            got: buf.len(),
        });
    }
    let layer_name = std::str::from_utf8(&buf[28..name_end])
        .map_err(|_| {
            LqerLoadError::InvalidName(String::from_utf8(buf[28..name_end].to_vec()).unwrap_err())
        })?
        .to_string();

    // 2026-09-26: The matrices start at the first multiple of 8 at or after
    // the end of the name.
    let aligned = (name_end + 7) & !7;
    if buf.len() < aligned {
        return Err(LqerLoadError::Truncated {
            expected: aligned,
            got: buf.len(),
        });
    }

    let left_bytes = rows
        .checked_mul(rank)
        .and_then(|x| x.checked_mul(2))
        .ok_or_else(|| LqerLoadError::ShapeMismatch("rows × rank × 2 overflowed".into()))?;
    let left_end = aligned + left_bytes;
    if buf.len() < left_end {
        return Err(LqerLoadError::Truncated {
            expected: left_end,
            got: buf.len(),
        });
    }
    let left_bf16 = buf[aligned..left_end].to_vec();

    let right_bytes = rank
        .checked_mul(cols)
        .and_then(|x| x.checked_mul(2))
        .ok_or_else(|| LqerLoadError::ShapeMismatch("rank × cols × 2 overflowed".into()))?;
    let right_end = left_end + right_bytes;
    if buf.len() < right_end {
        return Err(LqerLoadError::Truncated {
            expected: right_end,
            got: buf.len(),
        });
    }
    let right_bf16 = buf[left_end..right_end].to_vec();

    let c = LqerCorrection {
        layer_name,
        rank,
        rows,
        cols,
        left_bf16,
        right_bf16,
    };
    c.validate().map_err(LqerLoadError::ShapeMismatch)?;
    Ok(c)
}

/// 2026-09-26: Serialise to the `.lqer` format; `Err` when the correction
/// fails `validate`. Only the tests call it.
pub fn write_lqer_bytes(c: &LqerCorrection) -> Result<Vec<u8>, LqerLoadError> {
    c.validate().map_err(LqerLoadError::ShapeMismatch)?;
    let name_bytes = c.layer_name.as_bytes();
    let header_end = 28 + name_bytes.len();
    let aligned = (header_end + 7) & !7;
    let total = aligned + c.left_bf16.len() + c.right_bf16.len();
    let mut buf = Vec::with_capacity(total);
    buf.extend_from_slice(LQER_MAGIC);
    buf.extend_from_slice(&LQER_VERSION.to_le_bytes());
    buf.extend_from_slice(&(c.rank as u32).to_le_bytes());
    buf.extend_from_slice(&(c.rows as u32).to_le_bytes());
    buf.extend_from_slice(&(c.cols as u32).to_le_bytes());
    buf.extend_from_slice(&(name_bytes.len() as u32).to_le_bytes());
    buf.extend_from_slice(name_bytes);
    while buf.len() < aligned {
        buf.push(0);
    }
    buf.extend_from_slice(&c.left_bf16);
    buf.extend_from_slice(&c.right_bf16);
    Ok(buf)
}

/// 2026-09-26: Load every `.lqer` file in `dir`. An unreadable directory
/// gives an empty set; a file that cannot be read or parsed is logged and
/// skipped.
pub fn load_from_dir(dir: &Path) -> LqerCorrectionSet {
    let mut set = LqerCorrectionSet::empty();
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return set,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("lqer") {
            continue;
        }
        match std::fs::read(&path)
            .map_err(LqerLoadError::from)
            .and_then(|b| parse_lqer_bytes(&b))
        {
            Ok(c) => set.insert(c),
            Err(e) => {
                tracing::warn!("Skipping malformed LQER file {}: {}", path.display(), e);
            }
        }
    }
    if !set.is_empty() {
        tracing::info!(
            count = set.len(),
            mem_mb = set.total_memory_bytes() / 1_048_576,
            "Loaded LQER corrections from {}",
            dir.display()
        );
    }
    set
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy(rows: usize, cols: usize, rank: usize) -> LqerCorrection {
        LqerCorrection {
            layer_name: "test".into(),
            rank,
            rows,
            cols,
            left_bf16: vec![0u8; rows * rank * 2],
            right_bf16: vec![0u8; rank * cols * 2],
        }
    }

    #[test]
    fn validate_passes_on_correct_shape() {
        let c = dummy(64, 128, 8);
        assert!(c.validate().is_ok());
    }

    #[test]
    fn validate_fails_on_wrong_left_shape() {
        let mut c = dummy(64, 128, 8);
        c.left_bf16.truncate(10);
        let err = c.validate().unwrap_err();
        assert!(err.contains("left matrix size mismatch"));
    }

    #[test]
    fn memory_bytes_is_sum_of_matrices() {
        let c = dummy(64, 128, 8);
        assert_eq!(c.memory_bytes(), 3072);
    }

    #[test]
    fn correction_set_round_trip() {
        let mut s = LqerCorrectionSet::empty();
        assert!(s.is_empty());
        let mut c = dummy(64, 128, 8);
        c.layer_name = "l0".into();
        s.insert(c);
        assert_eq!(s.len(), 1);
        assert!(s.get("l0").is_some());
        assert!(s.get("l1").is_none());
    }

    #[test]
    fn suggest_rank_at_various_fractions() {
        assert_eq!(suggest_rank(2048, 6144, 0.10), 205);
        assert_eq!(suggest_rank(2048, 6144, 0.30), 615);
        assert_eq!(suggest_rank(100, 100, 0.0), 1);
        assert_eq!(suggest_rank(100, 100, 1.5), 100);
    }

    fn make_correction(name: &str, rows: usize, cols: usize, rank: usize) -> LqerCorrection {
        let left_bf16: Vec<u8> = (0..rows * rank * 2).map(|i| (i & 0xFF) as u8).collect();
        let right_bf16: Vec<u8> = (0..rank * cols * 2)
            .map(|i| ((i ^ 0xA5) & 0xFF) as u8)
            .collect();
        LqerCorrection {
            layer_name: name.to_string(),
            rank,
            rows,
            cols,
            left_bf16,
            right_bf16,
        }
    }

    #[test]
    fn write_then_parse_round_trips() {
        let original = make_correction("model.layers.0.mlp.experts.5.down_proj", 64, 128, 8);
        let bytes = write_lqer_bytes(&original).expect("serialise");
        let parsed = parse_lqer_bytes(&bytes).expect("parse");
        assert_eq!(parsed.layer_name, original.layer_name);
        assert_eq!(parsed.rank, original.rank);
        assert_eq!(parsed.rows, original.rows);
        assert_eq!(parsed.cols, original.cols);
        assert_eq!(parsed.left_bf16, original.left_bf16);
        assert_eq!(parsed.right_bf16, original.right_bf16);
    }

    #[test]
    fn parse_rejects_bad_magic() {
        let mut bytes = write_lqer_bytes(&make_correction("x", 8, 8, 2)).unwrap();
        bytes[0] = b'X';
        let err = parse_lqer_bytes(&bytes).unwrap_err();
        assert!(matches!(err, LqerLoadError::BadMagic));
    }

    #[test]
    fn parse_rejects_unsupported_version() {
        let mut bytes = write_lqer_bytes(&make_correction("x", 8, 8, 2)).unwrap();
        bytes[8] = 99;
        let err = parse_lqer_bytes(&bytes).unwrap_err();
        assert!(matches!(err, LqerLoadError::UnsupportedVersion(_)));
    }

    #[test]
    fn parse_rejects_truncated() {
        let bytes = write_lqer_bytes(&make_correction("x", 8, 8, 2)).unwrap();
        let truncated = &bytes[..bytes.len() - 4];
        let err = parse_lqer_bytes(truncated).unwrap_err();
        assert!(matches!(err, LqerLoadError::Truncated { .. }));
    }

    #[test]
    fn load_from_missing_dir_returns_empty() {
        let set = load_from_dir(Path::new("/nonexistent/path/should/not/exist"));
        assert!(set.is_empty());
    }

    #[test]
    fn load_from_dir_reads_multiple_lqer_files() {
        let tmp = std::env::temp_dir().join(format!(
            "metrale_lqer_test_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&tmp).unwrap();
        let a = make_correction("a", 16, 16, 4);
        let b = make_correction("b", 32, 16, 4);
        std::fs::write(tmp.join("a.lqer"), write_lqer_bytes(&a).unwrap()).unwrap();
        std::fs::write(tmp.join("b.lqer"), write_lqer_bytes(&b).unwrap()).unwrap();
        std::fs::write(tmp.join("garbage.lqer"), b"not a real file").unwrap();
        std::fs::write(tmp.join("readme.txt"), b"hello").unwrap();
        let set = load_from_dir(&tmp);
        assert_eq!(set.len(), 2, "two valid files load, garbage skipped");
        assert!(set.get("a").is_some());
        assert!(set.get("b").is_some());
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn write_layer_name_with_dots_round_trips() {
        let c = make_correction("model.layers.10.mlp.experts.0.down_proj.weight", 32, 64, 4);
        let bytes = write_lqer_bytes(&c).unwrap();
        let parsed = parse_lqer_bytes(&bytes).unwrap();
        assert_eq!(parsed.layer_name, c.layer_name);
    }
}
