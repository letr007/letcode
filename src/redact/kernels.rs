use anyhow::{Result, bail};

#[cfg(target_arch = "x86_64")]
const TILE_COLS: usize = 8;
#[cfg(target_arch = "x86_64")]
const TILE_ROWS: usize = 2;

#[cfg(target_arch = "aarch64")]
const TILE_COLS: usize = 4;
#[cfg(target_arch = "aarch64")]
const TILE_ROWS: usize = 4;

pub(crate) struct QMatrix {
    pub(crate) k: usize,
    pub(crate) cols: usize,
    #[cfg(any(not(target_arch = "aarch64"), test))]
    w: Vec<i8>,
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    blocked: Vec<i8>,
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    k_groups: usize,
    pub(crate) scale: Vec<f32>,
    pub(crate) zp: Vec<i32>,
    col_sum: Vec<i32>,
}

impl QMatrix {
    pub(crate) fn new(
        k: usize,
        cols: usize,
        w: Vec<i8>,
        scale: Vec<f32>,
        zp: Vec<i32>,
    ) -> Result<Self> {
        if w.len() != k * cols || scale.len() != cols || zp.len() != cols {
            bail!(
                "a weight matrix of {k}x{cols} came with {} values, {} scales and {} zero points",
                w.len(),
                scale.len(),
                zp.len()
            );
        }
        let mut col_sum = vec![0i32; cols];
        for row in 0..k {
            for (c, sum) in col_sum.iter_mut().enumerate() {
                *sum += i32::from(w[row * cols + c]);
            }
        }
        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
        let k_groups = k.div_ceil(TILE_ROWS);
        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
        let mut blocked = vec![0i8; cols.div_ceil(TILE_COLS) * k_groups * TILE_COLS * TILE_ROWS];
        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
        for tile in 0..cols.div_ceil(TILE_COLS) {
            for group in 0..k_groups {
                let base = (tile * k_groups + group) * TILE_COLS * TILE_ROWS;
                for c in 0..TILE_COLS {
                    let col = tile * TILE_COLS + c;
                    for r in 0..TILE_ROWS {
                        let row = group * TILE_ROWS + r;
                        if col < cols && row < k {
                            blocked[base + c * TILE_ROWS + r] = w[row * cols + col];
                        }
                    }
                }
            }
        }
        Ok(Self {
            k,
            cols,
            #[cfg(any(not(target_arch = "aarch64"), test))]
            w,
            #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
            blocked,
            #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
            k_groups,
            scale,
            zp,
            col_sum,
        })
    }

    pub(crate) fn gemm(
        &self,
        a: &[u8],
        rows: usize,
        a_scale: f32,
        azp: i32,
        bias: Option<&[f32]>,
        out: &mut [f32],
    ) {
        debug_assert_eq!(a.len(), rows * self.k);
        debug_assert_eq!(out.len(), rows * self.cols);
        let cols = self.cols;
        let work = |r0: usize, r1: usize, out: &mut [f32], acc: &mut Vec<i32>| {
            for r in r0..r1 {
                let row = &a[r * self.k..(r + 1) * self.k];
                let sum_a: i32 = row.iter().map(|&v| i32::from(v)).sum();
                self.kernel(row, acc);
                let dst = &mut out[(r - r0) * cols..(r - r0 + 1) * cols];
                for c in 0..cols {
                    let v = acc[c] - azp * self.col_sum[c] - self.zp[c] * sum_a
                        + self.k as i32 * azp * self.zp[c];
                    dst[c] = v as f32 * a_scale * self.scale[c] + bias.map_or(0.0, |b| b[c]);
                }
            }
        };
        let chunk_rows = rows.div_ceil(threads(rows));
        std::thread::scope(|s| {
            for (index, piece) in out.chunks_mut(chunk_rows * cols).enumerate() {
                let r0 = index * chunk_rows;
                let r1 = r0 + piece.len() / cols;
                let work = &work;
                s.spawn(move || {
                    let mut acc = vec![0i32; cols];
                    work(r0, r1, piece, &mut acc);
                });
            }
        });
    }

    // VNNI would double this, but on a hybrid Intel CPU a thread scheduled onto
    // an efficiency core dies with SIGILL; AVX2 is on every core.
    fn kernel(&self, a: &[u8], out: &mut [i32]) {
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx2") {
                // SAFETY: AVX2 is present, and the blocked layout is built for
                // exactly these dimensions.
                unsafe { self.kernel_avx2(a, out) };
                return;
            }
            self.kernel_scalar(a, out);
        }
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("dotprod") {
                // SAFETY: dotprod is present, and the blocked layout holds one
                // 16-byte group per (tile, k group).
                unsafe { self.kernel_dotprod(a, out) };
            } else {
                // SAFETY: NEON is part of the aarch64 baseline.
                unsafe { self.kernel_neon(a, out) };
            }
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        self.kernel_scalar(a, out);
    }

    #[cfg(any(not(target_arch = "aarch64"), test))]
    fn kernel_scalar(&self, a: &[u8], out: &mut [i32]) {
        for (c, slot) in out.iter_mut().enumerate() {
            let mut acc = 0i32;
            for (x, w) in a.iter().zip(self.w.chunks_exact(self.cols)) {
                acc += i32::from(*x) * i32::from(w[c]);
            }
            *slot = acc;
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn kernel_avx2(&self, a: &[u8], out: &mut [i32]) {
        use std::arch::x86_64::*;

        let tiles = self.cols.div_ceil(8);
        let zero = _mm256_setzero_si256();
        let mut tile = 0;
        while tile < tiles {
            let group = (tiles - tile).min(4);
            let mut acc = [zero; 4];
            // SAFETY: the blocked layout holds one 16-byte group per
            // (tile, k-pair), and `a` has `k` values.
            unsafe {
                for pair in 0..self.k_groups {
                    let row = pair * 2;
                    let lo = i32::from(*a.get_unchecked(row));
                    let hi = if row + 1 < self.k {
                        i32::from(*a.get_unchecked(row + 1))
                    } else {
                        0
                    };
                    // One a value per 16-bit lane, alternating: the weight bytes
                    // are interleaved the same way, so `madd` sums a k pair.
                    let av = _mm256_set1_epi32(lo | (hi << 16));
                    for (g, slot) in acc.iter_mut().enumerate().take(group) {
                        let base = ((tile + g) * self.k_groups + pair) * 16;
                        let w = _mm_loadu_si128(self.blocked.as_ptr().add(base).cast());
                        *slot =
                            _mm256_add_epi32(*slot, _mm256_madd_epi16(av, _mm256_cvtepi8_epi16(w)));
                    }
                }
            }
            for (g, acc) in acc.iter().enumerate().take(group) {
                let mut buf = [0i32; 8];
                // SAFETY: `buf` holds eight i32, the width of one accumulator.
                unsafe { _mm256_storeu_si256(buf.as_mut_ptr().cast(), *acc) };
                let col = (tile + g) * 8;
                let n = (self.cols - col).min(8);
                out[col..col + n].copy_from_slice(&buf[..n]);
            }
            tile += 4;
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    unsafe fn kernel_neon(&self, a: &[u8], out: &mut [i32]) {
        use std::arch::aarch64::*;

        let zero = vdupq_n_s32(0);
        let tiles = self.cols.div_ceil(TILE_COLS);
        let mut tile = 0;
        while tile < tiles {
            let group = (tiles - tile).min(4);
            let mut acc = [(zero, zero); 4];
            for k_group in 0..self.k_groups {
                let start = k_group * TILE_ROWS;
                let end = (start + TILE_ROWS).min(self.k);
                let mut bytes = [0u8; 4];
                bytes[..end - start].copy_from_slice(&a[start..end]);
                // The activations are uint8 and the multiply is signed, so they
                // enter centered and the store adds the bias back.
                let av = vreinterpret_s8_u32(vdup_n_u32(u32::from_le_bytes(bytes) ^ 0x8080_8080));
                for (g, (lo, hi)) in acc.iter_mut().enumerate().take(group) {
                    // SAFETY: the blocked layout holds one 16-byte group per
                    // (tile, k group), and `a` has `k` values.
                    let w = unsafe {
                        vld1q_s8(
                            self.blocked
                                .as_ptr()
                                .add(((tile + g) * self.k_groups + k_group) * 16),
                        )
                    };
                    *lo = vpadalq_s16(*lo, vmull_s8(vget_low_s8(w), av));
                    *hi = vpadalq_s16(*hi, vmull_s8(vget_high_s8(w), av));
                }
            }
            for (g, (lo, hi)) in acc.iter().enumerate().take(group) {
                let mut sums = [0i32; 4];
                // SAFETY: `sums` holds four i32, the width of one accumulator pair.
                unsafe { vst1q_s32(sums.as_mut_ptr(), vpaddq_s32(*lo, *hi)) };
                let col = (tile + g) * TILE_COLS;
                let n = (self.cols - col).min(TILE_COLS);
                for c in 0..n {
                    out[col + c] = sums[c] + 128 * self.col_sum[col + c];
                }
            }
            tile += 4;
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon,dotprod")]
    unsafe fn kernel_dotprod(&self, a: &[u8], out: &mut [i32]) {
        use std::arch::aarch64::*;

        const TILES: usize = 16;
        let zero = vdupq_n_s32(0);
        let tiles = self.cols.div_ceil(TILE_COLS);
        let mut tile = 0;
        while tile < tiles {
            let group = (tiles - tile).min(TILES);
            let mut acc = [zero; TILES];
            for k_group in 0..self.k_groups {
                let start = k_group * TILE_ROWS;
                let end = (start + TILE_ROWS).min(self.k);
                let mut bytes = [0u8; 4];
                bytes[..end - start].copy_from_slice(&a[start..end]);
                // Centered, as in the NEON kernel: the products are signed.
                let av = vreinterpretq_s8_u32(vdupq_n_u32(u32::from_le_bytes(bytes) ^ 0x8080_8080));
                for (g, slot) in acc.iter_mut().enumerate().take(group) {
                    // SAFETY: the blocked layout holds one 16-byte group per
                    // (tile, k group), and `a` has `k` values.
                    let w = unsafe {
                        vld1q_s8(
                            self.blocked
                                .as_ptr()
                                .add(((tile + g) * self.k_groups + k_group) * 16),
                        )
                    };
                    // SAFETY: `kernel_dotprod` carries the dotprod feature.
                    *slot = unsafe { dot(*slot, av, w) };
                }
            }
            for (g, acc) in acc.iter().enumerate().take(group) {
                let mut sums = [0i32; 4];
                // SAFETY: `sums` holds four i32, the width of the accumulator.
                unsafe { vst1q_s32(sums.as_mut_ptr(), *acc) };
                let col = (tile + g) * TILE_COLS;
                let n = (self.cols - col).min(TILE_COLS);
                for c in 0..n {
                    out[col + c] = sums[c] + 128 * self.col_sum[col + c];
                }
            }
            tile += TILES;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,dotprod")]
unsafe fn dot(
    acc: std::arch::aarch64::int32x4_t,
    a: std::arch::aarch64::int8x16_t,
    w: std::arch::aarch64::int8x16_t,
) -> std::arch::aarch64::int32x4_t {
    let mut out = acc;
    // `vdotq_s32` is not stable yet, so the dot product is emitted directly.
    // SAFETY: the caller carries the dotprod feature; the instruction reads
    // only its two vector operands and writes the four accumulator lanes.
    unsafe {
        std::arch::asm!(
            "sdot {out:v}.4s, {a:v}.16b, {w:v}.16b",
            out = inout(vreg) out,
            a = in(vreg) a,
            w = in(vreg) w,
            options(pure, nomem, nostack),
        );
    }
    out
}

pub(crate) fn threads(rows: usize) -> usize {
    let cores = std::thread::available_parallelism().map_or(1, |n| n.get());
    (cores / 2).clamp(1, 8).min((rows / 32).max(1))
}

// Scale covers the range through zero; the zero point rounds half to even.
pub(crate) fn quantize_into(out: &mut Vec<u8>, x: &[f32]) -> (f32, i32) {
    let mut min = 0.0f32;
    let mut max = 0.0f32;
    for &v in x {
        if v < min {
            min = v;
        }
        if v > max {
            max = v;
        }
    }
    let scale = (max - min) / 255.0;
    out.clear();
    if scale <= 0.0 || !scale.is_finite() {
        out.resize(x.len(), 0);
        return (1.0, 0);
    }
    let zp = (-min / scale).round_ties_even().clamp(0.0, 255.0) as i32;
    out.extend(x.iter().map(|&v| {
        let q = (v / scale).round_ties_even() + zp as f32;
        q.clamp(0.0, 255.0) as u8
    }));
    (scale, zp)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_packed_kernel_matches_the_scalar_one() {
        let (k, cols) = (18, 21);
        let mut state = 0x2545_f491u32;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state % 255) as i32 - 127
        };
        let w: Vec<i8> = (0..k * cols).map(|_| next() as i8).collect();
        let matrix =
            QMatrix::new(k, cols, w, vec![0.5; cols], vec![0; cols]).expect("the matrix builds");
        for row in 0..3u8 {
            let a: Vec<u8> = (0..k)
                .map(|i| (i as u8).wrapping_mul(37).wrapping_add(row))
                .collect();
            let mut want = vec![0i32; cols];
            matrix.kernel_scalar(&a, &mut want);
            let mut got = vec![0i32; cols];
            matrix.kernel(&a, &mut got);
            assert_eq!(want, got);
        }
    }
}
