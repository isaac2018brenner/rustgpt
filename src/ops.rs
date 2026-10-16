//! The building blocks of a transformer, each with a hand-derived backward pass.
//!
//! Layout convention: activations are row-major `[N, C]` where `N = batch * time`.
//! Backward functions either overwrite their input gradient (`dinp = ...`) or
//! accumulate into it (`dinp += ...`); each doc comment says which.

use crate::parallel::par_each;

/// Rows handed to a worker thread at a time.
const ROW_BLOCK: usize = 16;
/// Rows of the input kept hot in cache while accumulating weight gradients.
const CACHE_BLOCK: usize = 64;

/// Dot product with 8 independent accumulators so the compiler can vectorize it.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let ca = a.chunks_exact(8);
    let cb = b.chunks_exact(8);
    let (ra, rb) = (ca.remainder(), cb.remainder());
    let mut acc = [0.0f32; 8];
    for (x, y) in ca.zip(cb) {
        for k in 0..8 {
            acc[k] += x[k] * y[k];
        }
    }
    let mut s: f32 = acc.iter().sum();
    for (x, y) in ra.iter().zip(rb) {
        s += x * y;
    }
    s
}

/// `y += a * x`
#[inline]
pub fn axpy(y: &mut [f32], a: f32, x: &[f32]) {
    for (yi, xi) in y.iter_mut().zip(x) {
        *yi += a * xi;
    }
}

// ---------------------------------------------------------------------------
// Linear layer
// ---------------------------------------------------------------------------

/// `out[N, oc] = inp[N, c] @ w[oc, c]^T + bias[oc]`
pub fn matmul_forward(out: &mut [f32], inp: &[f32], w: &[f32], bias: Option<&[f32]>, c: usize, oc: usize) {
    let items: Vec<_> = out.chunks_mut(ROW_BLOCK * oc).zip(inp.chunks(ROW_BLOCK * c)).collect();
    par_each(items, |(out, inp)| {
        // Loop over weight rows on the outside so each one is reused across the block.
        for j in 0..oc {
            let wrow = &w[j * c..(j + 1) * c];
            let bj = bias.map_or(0.0, |b| b[j]);
            for (orow, xrow) in out.chunks_mut(oc).zip(inp.chunks(c)) {
                orow[j] = dot(xrow, wrow) + bj;
            }
        }
    });
}

/// `dinp = dout @ w` (overwrite), `dw += dout^T @ inp`, `dbias += sum(dout)`.
#[allow(clippy::too_many_arguments)]
pub fn matmul_backward(
    dinp: &mut [f32],
    dw: &mut [f32],
    dbias: Option<&mut [f32]>,
    dout: &[f32],
    inp: &[f32],
    w: &[f32],
    c: usize,
    oc: usize,
) {
    let n = inp.len() / c;

    let items: Vec<_> = dinp.chunks_mut(ROW_BLOCK * c).zip(dout.chunks(ROW_BLOCK * oc)).collect();
    par_each(items, |(dinp, dout)| {
        for (dirow, drow) in dinp.chunks_mut(c).zip(dout.chunks(oc)) {
            dirow.fill(0.0);
            for (j, &g) in drow.iter().enumerate() {
                axpy(dirow, g, &w[j * c..(j + 1) * c]);
            }
        }
    });

    let items: Vec<_> = dw.chunks_mut(ROW_BLOCK * c).enumerate().collect();
    par_each(items, |(blk, dw)| {
        for r0 in (0..n).step_by(CACHE_BLOCK) {
            let r1 = (r0 + CACHE_BLOCK).min(n);
            for (jj, dwrow) in dw.chunks_mut(c).enumerate() {
                let j = blk * ROW_BLOCK + jj;
                for r in r0..r1 {
                    axpy(dwrow, dout[r * oc + j], &inp[r * c..(r + 1) * c]);
                }
            }
        }
    });

    if let Some(db) = dbias {
        for drow in dout.chunks(oc) {
            axpy(db, 1.0, drow);
        }
    }
}

// ---------------------------------------------------------------------------
// LayerNorm
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
pub fn layernorm_forward(
    out: &mut [f32],
    mean: &mut [f32],
    rstd: &mut [f32],
    inp: &[f32],
    w: &[f32],
    b: &[f32],
    c: usize,
) {
    for (i, (orow, xrow)) in out.chunks_mut(c).zip(inp.chunks(c)).enumerate() {
        let m = xrow.iter().sum::<f32>() / c as f32;
        let var = xrow.iter().map(|x| (x - m) * (x - m)).sum::<f32>() / c as f32;
        let s = 1.0 / (var + 1e-5).sqrt();
        for j in 0..c {
            orow[j] = (xrow[j] - m) * s * w[j] + b[j];
        }
        mean[i] = m;
        rstd[i] = s;
    }
}

/// `dinp += ...`, `dw += ...`, `db += ...`
#[allow(clippy::too_many_arguments)]
pub fn layernorm_backward(
    dinp: &mut [f32],
    dw: &mut [f32],
    db: &mut [f32],
    dout: &[f32],
    inp: &[f32],
    w: &[f32],
    mean: &[f32],
    rstd: &[f32],
    c: usize,
) {
    for (i, ((dirow, drow), xrow)) in dinp.chunks_mut(c).zip(dout.chunks(c)).zip(inp.chunks(c)).enumerate() {
        let (m, s) = (mean[i], rstd[i]);
        // Two reductions needed by the normalization's Jacobian.
        let mut dnorm_mean = 0.0;
        let mut dnorm_norm_mean = 0.0;
        for j in 0..c {
            let norm = (xrow[j] - m) * s;
            let dnorm = w[j] * drow[j];
            dnorm_mean += dnorm;
            dnorm_norm_mean += dnorm * norm;
        }
        dnorm_mean /= c as f32;
        dnorm_norm_mean /= c as f32;
        for j in 0..c {
            let norm = (xrow[j] - m) * s;
            let dnorm = w[j] * drow[j];
            db[j] += drow[j];
            dw[j] += norm * drow[j];
            dirow[j] += (dnorm - dnorm_mean - norm * dnorm_norm_mean) * s;
        }
    }
}

// ---------------------------------------------------------------------------
// Causal multi-head self-attention
// ---------------------------------------------------------------------------

/// `qkv` is `[B, T, 3C]` (queries, keys, values concatenated). Writes the
/// attention-weighted values to `out` `[B, T, C]` and the post-softmax
/// attention matrix to `att` `[B, NH, T, T]` (needed for backward).
pub fn attention_forward(out: &mut [f32], att: &mut [f32], qkv: &[f32], t: usize, c: usize, nh: usize) {
    let hs = c / nh;
    let c3 = 3 * c;
    let scale = 1.0 / (hs as f32).sqrt();
    let items: Vec<_> = out.chunks_mut(t * c).zip(att.chunks_mut(nh * t * t)).zip(qkv.chunks(t * c3)).collect();
    par_each(items, |((out, att), x)| {
        for h in 0..nh {
            for t1 in 0..t {
                let q = &x[t1 * c3 + h * hs..][..hs];
                let arow = &mut att[(h * t + t1) * t..][..t];

                let mut maxv = f32::NEG_INFINITY;
                for (t2, a) in arow.iter_mut().enumerate().take(t1 + 1) {
                    let k = &x[t2 * c3 + c + h * hs..][..hs];
                    *a = dot(q, k) * scale;
                    maxv = maxv.max(*a);
                }
                let mut sum = 0.0;
                for a in &mut arow[..=t1] {
                    *a = (*a - maxv).exp();
                    sum += *a;
                }
                let inv = 1.0 / sum;
                arow[..=t1].iter_mut().for_each(|a| *a *= inv);
                arow[t1 + 1..].fill(0.0); // causal mask: no peeking at the future

                let orow = &mut out[t1 * c + h * hs..][..hs];
                orow.fill(0.0);
                for (t2, &a) in arow.iter().enumerate().take(t1 + 1) {
                    axpy(orow, a, &x[t2 * c3 + 2 * c + h * hs..][..hs]);
                }
            }
        }
    });
}

/// `dqkv = ...` (overwrite)
pub fn attention_backward(dqkv: &mut [f32], dout: &[f32], qkv: &[f32], att: &[f32], t: usize, c: usize, nh: usize) {
    let hs = c / nh;
    let c3 = 3 * c;
    let scale = 1.0 / (hs as f32).sqrt();
    let items: Vec<_> =
        dqkv.chunks_mut(t * c3).zip(dout.chunks(t * c)).zip(qkv.chunks(t * c3)).zip(att.chunks(nh * t * t)).collect();
    par_each(items, |(((dx, dout), x), att)| {
        dx.fill(0.0);
        let mut datt = vec![0.0f32; t];
        for h in 0..nh {
            for t1 in 0..t {
                let arow = &att[(h * t + t1) * t..][..t];
                let drow = &dout[t1 * c + h * hs..][..hs];

                // Backward through the weighted sum of values.
                for t2 in 0..=t1 {
                    datt[t2] = dot(drow, &x[t2 * c3 + 2 * c + h * hs..][..hs]);
                    axpy(&mut dx[t2 * c3 + 2 * c + h * hs..][..hs], arow[t2], drow);
                }

                // Backward through softmax: dpre_i = a_i * (datt_i - sum_j a_j * datt_j).
                let s: f32 = (0..=t1).map(|i| arow[i] * datt[i]).sum();

                // Backward through q . k.
                for t2 in 0..=t1 {
                    let dpre = arow[t2] * (datt[t2] - s) * scale;
                    axpy(&mut dx[t1 * c3 + h * hs..][..hs], dpre, &x[t2 * c3 + c + h * hs..][..hs]);
                    axpy(&mut dx[t2 * c3 + c + h * hs..][..hs], dpre, &x[t1 * c3 + h * hs..][..hs]);
                }
            }
        }
    });
}

// ---------------------------------------------------------------------------
// Pointwise ops
// ---------------------------------------------------------------------------

const GELU_SCALE: f32 = 0.797_884_6; // sqrt(2 / pi)

/// GELU, tanh approximation (as in GPT-2).
pub fn gelu_forward(out: &mut [f32], inp: &[f32]) {
    for (o, &x) in out.iter_mut().zip(inp) {
        let cube = 0.044715 * x * x * x;
        *o = 0.5 * x * (1.0 + (GELU_SCALE * (x + cube)).tanh());
    }
}

/// `dinp = ...` (overwrite)
pub fn gelu_backward(dinp: &mut [f32], inp: &[f32], dout: &[f32]) {
    for ((di, &x), &d) in dinp.iter_mut().zip(inp).zip(dout) {
        let cube = 0.044715 * x * x * x;
        let th = (GELU_SCALE * (x + cube)).tanh();
        let sech2 = 1.0 - th * th;
        let grad = 0.5 * (1.0 + th) + 0.5 * x * sech2 * GELU_SCALE * (1.0 + 3.0 * 0.044715 * x * x);
        *di = grad * d;
    }
}

pub fn residual_forward(out: &mut [f32], a: &[f32], b: &[f32]) {
    for ((o, x), y) in out.iter_mut().zip(a).zip(b) {
        *o = x + y;
    }
}

/// Token embedding + learned position embedding.
pub fn encoder_forward(out: &mut [f32], tokens: &[usize], wte: &[f32], wpe: &[f32], t: usize, c: usize) {
    for (i, (orow, &tok)) in out.chunks_mut(c).zip(tokens).enumerate() {
        let pos = i % t;
        residual_forward(orow, &wte[tok * c..][..c], &wpe[pos * c..][..c]);
    }
}

/// `dwte += ...`, `dwpe += ...`
pub fn encoder_backward(dwte: &mut [f32], dwpe: &mut [f32], dout: &[f32], tokens: &[usize], t: usize, c: usize) {
    for (i, (drow, &tok)) in dout.chunks(c).zip(tokens).enumerate() {
        let pos = i % t;
        axpy(&mut dwte[tok * c..][..c], 1.0, drow);
        axpy(&mut dwpe[pos * c..][..c], 1.0, drow);
    }
}

/// Numerically stable softmax of one row.
pub fn softmax(out: &mut [f32], logits: &[f32]) {
    let maxv = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for (o, &l) in out.iter_mut().zip(logits) {
        *o = (l - maxv).exp();
        sum += *o;
    }
    out.iter_mut().for_each(|o| *o /= sum);
}
