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
        let work = |r0: usize, r1: usize, out: &mut [f32], acc: &mut [i32]| {
            let mut first = r0;
            while first + 4 <= r1 {
                let block = &a[first * self.k..(first + 4) * self.k];
                self.kernel_rows(block, acc);
                for r in 0..4 {
                    self.dequant_row(
                        &block[r * self.k..(r + 1) * self.k],
                        &acc[r * cols..(r + 1) * cols],
                        a_scale,
                        azp,
                        bias,
                        &mut out[(first + r - r0) * cols..(first + r - r0 + 1) * cols],
                    );
                }
                first += 4;
            }
            for r in first..r1 {
                let row = &a[r * self.k..(r + 1) * self.k];
                self.kernel(row, &mut acc[..cols]);
                self.dequant_row(
                    row,
                    &acc[..cols],
                    a_scale,
                    azp,
                    bias,
                    &mut out[(r - r0) * cols..(r - r0 + 1) * cols],
                );
            }
        };
        let workers = threads(rows);
        if workers == 1 {
            let mut acc = vec![0i32; 4 * cols];
            work(0, rows, out, &mut acc);
            return;
        }
        let chunk_rows = rows.div_ceil(workers);
        std::thread::scope(|s| {
            for (index, piece) in out.chunks_mut(chunk_rows * cols).enumerate() {
                let r0 = index * chunk_rows;
                let r1 = r0 + piece.len() / cols;
                let work = &work;
                s.spawn(move || {
                    let mut acc = vec![0i32; 4 * cols];
                    work(r0, r1, piece, &mut acc);
                });
            }
        });
    }

    fn dequant_row(
        &self,
        a: &[u8],
        acc: &[i32],
        a_scale: f32,
        azp: i32,
        bias: Option<&[f32]>,
        out: &mut [f32],
    ) {
        let sum_a: i32 = a.iter().map(|&v| i32::from(v)).sum();
        for c in 0..self.cols {
            let v = acc[c] - azp * self.col_sum[c] - self.zp[c] * sum_a
                + self.k as i32 * azp * self.zp[c];
            out[c] = v as f32 * a_scale * self.scale[c] + bias.map_or(0.0, |b| b[c]);
        }
    }

    fn kernel_rows(&self, a: &[u8], out: &mut [i32]) {
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx2")
                && self.k.is_multiple_of(2)
                && self.cols.is_multiple_of(16)
            {
                // SAFETY: AVX2 is present, four input rows and 16-column tiles fit without padding.
                unsafe { self.kernel_avx2_rows(a, out) };
                return;
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("dotprod")
                && self.k.is_multiple_of(4)
                && self.cols.is_multiple_of(16)
            {
                // SAFETY: dotprod is present, four input rows and 16-column tiles fit without padding.
                unsafe { self.kernel_dotprod_rows(a, out) };
                return;
            }
        }
        for r in 0..4 {
            self.kernel(
                &a[r * self.k..(r + 1) * self.k],
                &mut out[r * self.cols..(r + 1) * self.cols],
            );
        }
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
    unsafe fn kernel_avx2_rows(&self, a: &[u8], out: &mut [i32]) {
        use std::arch::x86_64::*;

        for tile in (0..self.cols / TILE_COLS).step_by(2) {
            let mut acc = [[_mm256_setzero_si256(); 4]; 2];
            for pair in 0..self.k_groups {
                let mut av = [_mm256_setzero_si256(); 4];
                for (r, slot) in av.iter_mut().enumerate() {
                    // SAFETY: each of the four rows has a full pair because k is even.
                    let (lo, hi) = unsafe {
                        let row = a.as_ptr().add(r * self.k + pair * 2);
                        (i32::from(*row), i32::from(*row.add(1)))
                    };
                    *slot = _mm256_set1_epi32(lo | (hi << 16));
                }
                for (g, column) in acc.iter_mut().enumerate() {
                    // SAFETY: cols is a multiple of 16, so both tiles are inside the blocked layout.
                    let w = unsafe {
                        _mm_loadu_si128(
                            self.blocked
                                .as_ptr()
                                .add(((tile + g) * self.k_groups + pair) * 16)
                                .cast(),
                        )
                    };
                    let w = _mm256_cvtepi8_epi16(w);
                    for (value, av) in column.iter_mut().zip(av) {
                        *value = _mm256_add_epi32(*value, _mm256_madd_epi16(av, w));
                    }
                }
            }
            for (g, column) in acc.iter().enumerate() {
                for (r, value) in column.iter().enumerate() {
                    // SAFETY: out holds four rows of cols, and each tile has eight complete columns.
                    unsafe {
                        _mm256_storeu_si256(
                            out.as_mut_ptr()
                                .add(r * self.cols + (tile + g) * TILE_COLS)
                                .cast(),
                            *value,
                        )
                    };
                }
            }
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
    unsafe fn kernel_dotprod_rows(&self, a: &[u8], out: &mut [i32]) {
        use std::arch::aarch64::*;

        for tile in (0..self.cols / TILE_COLS).step_by(4) {
            let mut acc = [[vdupq_n_s32(0); 4]; 4];
            for group in 0..self.k_groups {
                let mut av = [vdupq_n_s8(0); 4];
                for (r, slot) in av.iter_mut().enumerate() {
                    // SAFETY: `k` is a multiple of 4, so each of the four rows has a full 4-byte group at `group * 4`.
                    let bits = unsafe {
                        a.as_ptr()
                            .add(r * self.k + group * 4)
                            .cast::<u32>()
                            .read_unaligned()
                    };
                    *slot = vreinterpretq_s8_u32(vdupq_n_u32(bits.to_le() ^ 0x8080_8080));
                }
                for (g, column) in acc.iter_mut().enumerate() {
                    // SAFETY: `cols` is a multiple of 16, so this tile group lies inside the blocked layout.
                    let w = unsafe {
                        vld1q_s8(
                            self.blocked
                                .as_ptr()
                                .add(((tile + g) * self.k_groups + group) * 16),
                        )
                    };
                    for (value, av) in column.iter_mut().zip(av) {
                        // SAFETY: `kernel_dotprod_rows` carries the dotprod feature.
                        *value = unsafe { dot(*value, av, w) };
                    }
                }
            }
            for (g, column) in acc.iter().enumerate() {
                for (r, value) in column.iter().enumerate() {
                    let col = (tile + g) * TILE_COLS;
                    // SAFETY: `out` holds four rows of `cols`, and the tile group ends inside each row.
                    unsafe { vst1q_s32(out.as_mut_ptr().add(r * self.cols + col), *value) };
                    for c in 0..TILE_COLS {
                        out[r * self.cols + col + c] += 128 * self.col_sum[col + c];
                    }
                }
            }
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
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 is present; the kernel handles complete blocks and the remaining tail separately.
        return unsafe { quantize_avx2(out, x) };
    }
    quantize_scalar(out, x)
}

fn quantize_scalar(out: &mut Vec<u8>, x: &[f32]) -> (f32, i32) {
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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn quantize_avx2(out: &mut Vec<u8>, x: &[f32]) -> (f32, i32) {
    use std::arch::x86_64::*;

    let zero = _mm256_setzero_ps();
    let mut lo = zero;
    let mut hi = zero;
    let (blocks, tail) = x.as_chunks::<8>();
    for block in blocks {
        // SAFETY: each block contains eight readable floats.
        let v = unsafe { _mm256_loadu_ps(block.as_ptr()) };
        lo = _mm256_min_ps(v, lo);
        hi = _mm256_max_ps(v, hi);
    }
    let mut mins = [0.0f32; 8];
    let mut maxs = [0.0f32; 8];
    // SAFETY: both arrays hold eight floats, matching the vector width.
    unsafe {
        _mm256_storeu_ps(mins.as_mut_ptr(), lo);
        _mm256_storeu_ps(maxs.as_mut_ptr(), hi);
    }
    let min = mins.iter().chain(tail).copied().fold(0.0f32, f32::min);
    let max = maxs.iter().chain(tail).copied().fold(0.0f32, f32::max);
    let scale = (max - min) / 255.0;
    out.resize(x.len(), 0);
    if scale <= 0.0 || !scale.is_finite() {
        out.fill(0);
        return (1.0, 0);
    }
    let zp = (-min / scale).round_ties_even().clamp(0.0, 255.0) as i32;
    let scale8 = _mm256_set1_ps(scale);
    let zp8 = _mm256_set1_ps(zp as f32);
    let max8 = _mm256_set1_ps(255.0);
    for (input, output) in blocks.iter().zip(out.as_chunks_mut::<8>().0) {
        // SAFETY: input is an eight-float block.
        let v = unsafe { _mm256_loadu_ps(input.as_ptr()) };
        let q = _mm256_round_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(
            _mm256_div_ps(v, scale8),
        );
        let q = _mm256_add_ps(q, zp8);
        let q = _mm256_min_ps(_mm256_max_ps(q, zero), max8);
        let q = _mm256_cvttps_epi32(q);
        let q = _mm_packus_epi32(_mm256_castsi256_si128(q), _mm256_extracti128_si256::<1>(q));
        let q = _mm_packus_epi16(q, _mm_setzero_si128());
        // SAFETY: output holds the eight bytes written by storel.
        unsafe { _mm_storel_epi64(output.as_mut_ptr().cast(), q) };
    }
    for (&v, dst) in tail.iter().zip(&mut out[blocks.len() * 8..]) {
        let q = (v / scale).round_ties_even() + zp as f32;
        *dst = q.clamp(0.0, 255.0) as u8;
    }
    (scale, zp)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantization_matches_the_scalar_result() {
        let mut cases = vec![
            vec![],
            vec![0.0; 17],
            vec![-0.0; 17],
            vec![3.0; 17],
            vec![-3.0; 17],
            vec![f32::NAN; 17],
            vec![f32::INFINITY; 17],
            vec![f32::NEG_INFINITY; 17],
            vec![f32::MIN_POSITIVE; 17],
            vec![f32::from_bits(1); 17],
            vec![-f32::MAX, f32::MAX],
            vec![0.0, 255.0, 0.5, 1.5, 2.5, 253.5, 254.5, f32::NAN, -0.0],
            vec![-128.0, 127.0, -0.5, 0.5, 1.5, -1.5, -127.5, 126.5, f32::NAN],
        ];
        let mut state = 0x2545_f491u32;
        for len in [1, 7, 8, 9, 31, 2049] {
            let mut values = Vec::new();
            for _ in 0..len {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                values.push(f32::from_bits(state));
            }
            cases.push(values.clone());
            cases.push(values.into_iter().map(|v| v.sin() * 3.0).collect());
        }
        let mut want = vec![123; 4096];
        let mut got = want.clone();
        for values in cases {
            let (want_scale, want_zp) = quantize_scalar(&mut want, &values);
            let (got_scale, got_zp) = quantize_into(&mut got, &values);
            assert_eq!(want_scale.to_bits(), got_scale.to_bits());
            assert_eq!(want_zp, got_zp);
            assert_eq!(want, got);
        }
    }

    #[test]
    fn the_packed_kernel_matches_the_scalar_one() {
        for (k, cols, rows) in [
            (6, 16, 4),
            (8, 32, 5),
            (18, 21, 5),
            (7, 16, 5),
            (384, 32, 65),
        ] {
            let mut state = 0x2545_f491u32;
            let mut next = || {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                state as i8
            };
            let w: Vec<i8> = (0..k * cols).map(|_| next()).collect();
            let scale: Vec<f32> = (0..cols).map(|c| 0.25 + c as f32 * 0.03).collect();
            let zp: Vec<i32> = (0..cols).map(|c| c as i32 % 5 - 2).collect();
            let matrix = QMatrix::new(k, cols, w, scale, zp).expect("the matrix builds");
            let a: Vec<u8> = (0..rows * k)
                .map(|i| (i as u8).wrapping_mul(37).wrapping_add(11))
                .collect();

            let mut want_acc = vec![0i32; rows * cols];
            for r in 0..rows {
                matrix.kernel_scalar(
                    &a[r * k..(r + 1) * k],
                    &mut want_acc[r * cols..(r + 1) * cols],
                );
            }
            let mut got_acc = vec![0i32; 4 * cols];
            for first in (0..rows / 4 * 4).step_by(4) {
                matrix.kernel_rows(&a[first * k..(first + 4) * k], &mut got_acc);
                assert_eq!(
                    want_acc[first * cols..(first + 4) * cols],
                    got_acc[..4 * cols]
                );
            }
            for r in rows / 4 * 4..rows {
                matrix.kernel(&a[r * k..(r + 1) * k], &mut got_acc[..cols]);
                assert_eq!(want_acc[r * cols..(r + 1) * cols], got_acc[..cols]);
            }

            let (a_scale, azp) = (0.017, -3);
            let bias: Vec<f32> = (0..cols).map(|c| c as f32 * 0.01).collect();
            let mut got = vec![0f32; rows * cols];
            matrix.gemm(&a, rows, a_scale, azp, Some(&bias), &mut got);
            let mut want = vec![0f32; rows * cols];
            for r in 0..rows {
                let sum_a: i32 = a[r * k..(r + 1) * k].iter().map(|&v| i32::from(v)).sum();
                for c in 0..cols {
                    let v = want_acc[r * cols + c] - azp * matrix.col_sum[c] - matrix.zp[c] * sum_a
                        + k as i32 * azp * matrix.zp[c];
                    want[r * cols + c] = v as f32 * a_scale * matrix.scale[c] + bias[c];
                }
            }
            let want_bits: Vec<u32> = want.iter().map(|v| v.to_bits()).collect();
            let got_bits: Vec<u32> = got.iter().map(|v| v.to_bits()).collect();
            assert_eq!(want_bits, got_bits);
        }
    }
}
