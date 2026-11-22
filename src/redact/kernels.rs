use anyhow::{Result, bail};

pub(crate) struct QMatrix {
    pub(crate) k: usize,
    pub(crate) cols: usize,
    w: Vec<i8>,
    blocked: Vec<i8>,
    k_pairs: usize,
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
        let k_pairs = k.div_ceil(2);
        let mut blocked = vec![0i8; cols.div_ceil(8) * k_pairs * 16];
        for tile in 0..cols.div_ceil(8) {
            for pair in 0..k_pairs {
                let base = (tile * k_pairs + pair) * 16;
                for c in 0..8 {
                    let col = tile * 8 + c;
                    for half in 0..2 {
                        let row = pair * 2 + half;
                        if col < cols && row < k {
                            blocked[base + c * 2 + half] = w[row * cols + col];
                        }
                    }
                }
            }
        }
        Ok(Self {
            k,
            cols,
            w,
            blocked,
            k_pairs,
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
        }
        self.kernel_scalar(a, out);
    }

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
                for pair in 0..self.k_pairs {
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
                        let base = ((tile + g) * self.k_pairs + pair) * 16;
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
