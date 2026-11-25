// The export uses Microsoft's fused attention, skip-layer norm and bias GELU ops.
use anyhow::{Context, Result, bail};

use super::kernels::{QMatrix, quantize_into, threads};
use super::onnx::Graph;

pub(crate) const HIDDEN: usize = 384;
pub(crate) const HEADS: usize = 12;
pub(crate) const LAYERS: usize = 12;
pub(crate) const INTERMEDIATE: usize = 1536;
pub(crate) const CLASSES: usize = 21;
const HEAD_DIM: usize = HIDDEN / HEADS;
const EPSILON: f32 = 1e-12;

pub(crate) struct Embedding {
    q: Vec<u8>,
    scale: Vec<f32>,
    zp: Vec<u8>,
    dim: usize,
}

pub(crate) struct Layer {
    qkv: QMatrix,
    qkv_bias: Vec<f32>,
    out: QMatrix,
    out_bias: Vec<f32>,
    up: QMatrix,
    up_bias: Vec<f32>,
    smooth: Vec<f32>,
    down: QMatrix,
    down_bias: Vec<f32>,
    ln_attn: (Vec<f32>, Vec<f32>),
    ln_out: (Vec<f32>, Vec<f32>),
}

pub(crate) struct Weights {
    word: Embedding,
    position: Embedding,
    token_type: Embedding,
    ln_embed: (Vec<f32>, Vec<f32>),
    layers: Vec<Layer>,
    class: QMatrix,
    class_bias: Vec<f32>,
}

impl Weights {
    pub(crate) fn load(g: &Graph) -> Result<Self> {
        let word = embedding(g, "inner.bert.embeddings.word_embeddings.weight", HIDDEN)?;
        let position = embedding(
            g,
            "inner.bert.embeddings.position_embeddings.weight",
            HIDDEN,
        )?;
        let token_type = embedding(
            g,
            "inner.bert.embeddings.token_type_embeddings.weight",
            HIDDEN,
        )?;
        let ln_embed = (
            g.f32("inner.bert.embeddings.LayerNorm.weight")?,
            g.f32("inner.bert.embeddings.LayerNorm.bias")?,
        );

        let mut layers = Vec::with_capacity(LAYERS);
        for i in 0..LAYERS {
            let prefix = format!("inner.bert.encoder.layer.{i}");
            let qkv = weight(g, &format!("Attention_{i}_qkv_weight"), HIDDEN, HIDDEN * 3)?;
            let qkv_bias = g.f32(&format!("Attention_{i}_qkv_bias"))?;
            let out = matmul_weight(g, &prefix, "attention/output/dense")?;
            let up = matmul_weight(g, &prefix, "intermediate/dense")?;
            let down = matmul_weight(g, &prefix, "output/dense")?;
            layers.push(Layer {
                qkv,
                qkv_bias,
                out,
                out_bias: g.f32(&format!("{prefix}.attention.output.dense.bias"))?,
                up,
                up_bias: g.f32(&format!("{prefix}.intermediate.dense.bias"))?,
                smooth: g.f32(&format!("r15_smooth_inv_s_{i}"))?,
                down,
                down_bias: g.f32(&format!("{prefix}.output.dense.bias"))?,
                ln_attn: (
                    g.f32(&format!("{prefix}.attention.output.LayerNorm.weight"))?,
                    g.f32(&format!("{prefix}.attention.output.LayerNorm.bias"))?,
                ),
                ln_out: (
                    g.f32(&format!("{prefix}.output.LayerNorm.weight"))?,
                    g.f32(&format!("{prefix}.output.LayerNorm.bias"))?,
                ),
            });
        }

        let class_name = g
            .nodes
            .iter()
            .find(|n| n.op == "MatMulInteger" && n.name.contains("classifier"))
            .or_else(|| {
                g.nodes
                    .iter()
                    .find(|n| n.op == "MatMulInteger" && !n.name.contains("/inner/"))
            })
            .map(|n| n.inputs[1].clone())
            .context("the model has no classifier product")?;
        let class = weight_of(g, &class_name)?;
        let class_bias = g.f32("inner.classifier.bias")?;
        if class.cols != CLASSES || class.k != HIDDEN {
            bail!(
                "the classifier is {}x{}, not {}x{CLASSES}",
                class.k,
                class.cols,
                HIDDEN
            );
        }

        Ok(Self {
            word,
            position,
            token_type,
            ln_embed,
            layers,
            class,
            class_bias,
        })
    }

    pub(crate) fn logits(&self, ids: &[usize], types: &[usize]) -> Result<Vec<f32>> {
        let n = ids.len();
        let mut h = vec![0f32; n * HIDDEN];
        for i in 0..n {
            let word = ids[i]
                .checked_mul(HIDDEN)
                .context("a token id is past the vocabulary")?;
            if word + HIDDEN > self.word.q.len() {
                bail!("token {} is past the vocabulary", ids[i]);
            }
            let pos = i * HIDDEN;
            let typ = types[i].min(self.token_type.dim - 1) * HIDDEN;
            for c in 0..HIDDEN {
                h[i * HIDDEN + c] = self.word.at(word + c)
                    + self.position.at(pos + c)
                    + self.token_type.at(typ + c);
            }
        }
        layernorm_rows(&mut h, n, &self.ln_embed.0, &self.ln_embed.1);

        let mut qkv = vec![0f32; n * HIDDEN * 3];
        let mut attn = vec![0f32; n * HIDDEN];
        let mut heads = vec![0f32; n * HIDDEN * 3];
        let mut proj = vec![0f32; n * HIDDEN];
        let mut up = vec![0f32; n * INTERMEDIATE];
        let mut down = vec![0f32; n * HIDDEN];
        let mut quantized = Vec::with_capacity(n * INTERMEDIATE);

        for layer in &self.layers {
            let (scale, zp) = quantize_into(&mut quantized, &h);
            layer
                .qkv
                .gemm(&quantized, n, scale, zp, Some(&layer.qkv_bias), &mut qkv);
            attention(&qkv, n, &mut heads, &mut attn);

            let (scale, zp) = quantize_into(&mut quantized, &attn);
            layer.out.gemm(&quantized, n, scale, zp, None, &mut proj);
            skip_layernorm(
                &mut h,
                &proj,
                &layer.out_bias,
                &layer.ln_attn.0,
                &layer.ln_attn.1,
                n,
            );

            let (scale, zp) = quantize_into(&mut quantized, &h);
            layer.up.gemm(&quantized, n, scale, zp, None, &mut up);
            let activate = |up: &mut [f32]| {
                for row in up.chunks_mut(INTERMEDIATE) {
                    for (c, value) in row.iter_mut().enumerate() {
                        let x = *value + layer.up_bias[c];
                        *value = gelu(x) * layer.smooth[c];
                    }
                }
            };
            let workers = threads(n);
            if workers == 1 {
                activate(&mut up);
            } else {
                std::thread::scope(|s| {
                    for chunk in up.chunks_mut(n.div_ceil(workers) * INTERMEDIATE) {
                        let activate = &activate;
                        s.spawn(move || activate(chunk));
                    }
                });
            }

            let (scale, zp) = quantize_into(&mut quantized, &up);
            layer.down.gemm(&quantized, n, scale, zp, None, &mut down);
            skip_layernorm(
                &mut h,
                &down,
                &layer.down_bias,
                &layer.ln_out.0,
                &layer.ln_out.1,
                n,
            );
        }

        let (scale, zp) = quantize_into(&mut quantized, &h);
        let mut logits = vec![0f32; n * CLASSES];
        self.class.gemm(
            &quantized,
            n,
            scale,
            zp,
            Some(&self.class_bias),
            &mut logits,
        );
        Ok(logits)
    }
}

impl Embedding {
    fn at(&self, i: usize) -> f32 {
        (f32::from(self.q[i]) - f32::from(self.zp[i % HIDDEN])) * self.scale[i % HIDDEN]
    }
}

fn embedding(g: &Graph, name: &str, dim: usize) -> Result<Embedding> {
    let q = g.u8(&format!("{name}_quantized"))?;
    let scale = g.f32(&format!("{name}_scale"))?;
    let zp = g.u8(&format!("{name}_zero_point"))?;
    Ok(Embedding { q, scale, zp, dim })
}

fn weight(g: &Graph, name: &str, k: usize, cols: usize) -> Result<QMatrix> {
    weight_of(g, &format!("{name}_quantized")).and_then(|w| {
        if w.k == k && w.cols == cols {
            Ok(w)
        } else {
            bail!("{name} is not {k}x{cols}")
        }
    })
}

fn weight_of(g: &Graph, quantized: &str) -> Result<QMatrix> {
    let base = quantized.trim_end_matches("_quantized");
    let w = g.i8(quantized)?;
    let scale = g.f32(&format!("{base}_scale"))?;
    let zp = g.zp(&format!("{base}_zero_point"))?;
    let cols = scale.len();
    if cols == 0 || w.len() % cols != 0 {
        bail!("{quantized} has {} values over {cols} columns", w.len());
    }
    QMatrix::new(w.len() / cols, cols, w, scale, zp)
}

fn matmul_weight(g: &Graph, prefix: &str, path: &str) -> Result<QMatrix> {
    let layer = prefix.rsplit('.').next().context("a layer prefix")?;
    let scope = format!("layer.{layer}/{path}/");
    let node = g
        .nodes
        .iter()
        .find(|n| n.op == "MatMulInteger" && n.name.contains(&scope))
        .with_context(|| format!("the model has no {path} product in {prefix}"))?;
    weight_of(g, &node.inputs[1])
}

fn layernorm_rows(h: &mut [f32], rows: usize, gamma: &[f32], beta: &[f32]) {
    for r in 0..rows {
        let row = &mut h[r * HIDDEN..(r + 1) * HIDDEN];
        normalize(row, gamma, beta);
    }
}

fn skip_layernorm(
    h: &mut [f32],
    branch: &[f32],
    bias: &[f32],
    gamma: &[f32],
    beta: &[f32],
    rows: usize,
) {
    for r in 0..rows {
        let row = &mut h[r * HIDDEN..(r + 1) * HIDDEN];
        let b = &branch[r * HIDDEN..(r + 1) * HIDDEN];
        for c in 0..HIDDEN {
            row[c] += b[c] + bias[c];
        }
        normalize(row, gamma, beta);
    }
}

fn normalize(row: &mut [f32], gamma: &[f32], beta: &[f32]) {
    let n = row.len() as f32;
    let mean = row.iter().sum::<f32>() / n;
    let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
    let inv = 1.0 / (var + EPSILON).sqrt();
    for c in 0..row.len() {
        row[c] = (row[c] - mean) * inv * gamma[c] + beta[c];
    }
}

fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + erf(x * std::f32::consts::FRAC_1_SQRT_2))
}

// Abramowitz and Stegun 7.1.26, error under 1.5e-7.
fn erf(x: f32) -> f32 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let y = 1.0
        - (((((1.061_405_4 * t - 1.453_152_1) * t + 1.421_413_8) * t - 0.284_496_72) * t
            + 0.254_829_6)
            * t)
            * (-x * x).exp();
    sign * y
}

fn attention(qkv: &[f32], n: usize, heads: &mut [f32], out: &mut [f32]) {
    let stride = HIDDEN * 3;
    let head_stride = n * HEAD_DIM;
    let (q, kv) = heads.split_at_mut(n * HIDDEN);
    let (k, v) = kv.split_at_mut(n * HIDDEN);
    for head in 0..HEADS {
        let off = head * HEAD_DIM;
        for r in 0..n {
            let dst = head * head_stride + r * HEAD_DIM;
            q[dst..dst + HEAD_DIM].copy_from_slice(&qkv[r * stride + off..][..HEAD_DIM]);
            k[dst..dst + HEAD_DIM].copy_from_slice(&qkv[r * stride + HIDDEN + off..][..HEAD_DIM]);
            v[dst..dst + HEAD_DIM]
                .copy_from_slice(&qkv[r * stride + HIDDEN * 2 + off..][..HEAD_DIM]);
        }
    }

    let workers = threads(n);
    if workers == 1 {
        attend(q, k, v, n);
    } else {
        let chunk = HEADS.div_ceil(workers) * head_stride;
        std::thread::scope(|s| {
            for ((q, k), v) in q
                .chunks_mut(chunk)
                .zip(k.chunks(chunk))
                .zip(v.chunks(chunk))
            {
                s.spawn(move || attend(q, k, v, n));
            }
        });
    }

    for head in 0..HEADS {
        let off = head * HEAD_DIM;
        for r in 0..n {
            let src = head * head_stride + r * HEAD_DIM;
            out[r * HIDDEN + off..][..HEAD_DIM].copy_from_slice(&q[src..src + HEAD_DIM]);
        }
    }
}

fn attend(q: &mut [f32], k: &[f32], v: &[f32], n: usize) {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 is present; both implementations use the same slice bounds.
        unsafe { attend_avx2(q, k, v, n) };
        return;
    }
    attend_inner(q, k, v, n);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn attend_avx2(q: &mut [f32], k: &[f32], v: &[f32], n: usize) {
    attend_inner(q, k, v, n);
}

#[inline(always)]
fn attend_inner(q: &mut [f32], k: &[f32], v: &[f32], n: usize) {
    let head_stride = n * HEAD_DIM;
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let mut scores = vec![0f32; n];
    for ((q, k), v) in q
        .chunks_mut(head_stride)
        .zip(k.chunks(head_stride))
        .zip(v.chunks(head_stride))
    {
        for query in q.chunks_mut(HEAD_DIM) {
            let mut max = f32::NEG_INFINITY;
            for (j, slot) in scores.iter_mut().enumerate() {
                // SAFETY: each query and key contains HEAD_DIM floats.
                let dot = unsafe { dot32(query.as_ptr(), k.as_ptr().add(j * HEAD_DIM)) };
                let value = dot * scale;
                *slot = value;
                if value > max {
                    max = value;
                }
            }
            let mut sum = 0f32;
            for slot in &mut scores {
                *slot = (*slot - max).exp();
                sum += *slot;
            }
            let mut values = [0f32; HEAD_DIM];
            for (&p, value) in scores.iter().zip(v.as_chunks::<HEAD_DIM>().0) {
                let p = p / sum;
                for c in 0..HEAD_DIM {
                    values[c] += p * value[c];
                }
            }
            query.copy_from_slice(&values);
        }
    }
}

/// # Safety
/// `a` and `b` must each be readable for `HEAD_DIM` floats.
#[inline]
unsafe fn dot32(a: *const f32, b: *const f32) -> f32 {
    let mut acc = [0f32; 8];
    let mut d = 0;
    while d < HEAD_DIM {
        // SAFETY: the caller guarantees HEAD_DIM readable floats.
        unsafe {
            for (lane, slot) in acc.iter_mut().enumerate() {
                *slot += *a.add(d + lane) * *b.add(d + lane);
            }
        }
        d += 8;
    }
    acc.iter().sum()
}
