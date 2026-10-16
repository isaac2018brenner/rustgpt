//! A GPT-2 style decoder-only transformer with a fully manual forward and backward pass.

use crate::ops::*;
use crate::rng::Rng;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Config {
    pub vocab_size: usize,
    pub block_size: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_embd: usize,
}

pub const NUM_TENSORS: usize = 16;

pub const PARAM_NAMES: [&str; NUM_TENSORS] = [
    "wte", "wpe", "ln1w", "ln1b", "qkvw", "qkvb", "attprojw", "attprojb", "ln2w", "ln2b", "fcw", "fcb", "fcprojw",
    "fcprojb", "lnfw", "lnfb",
];

/// Whether AdamW applies weight decay to each tensor (matrices yes; biases and norms no).
pub const DECAY: [bool; NUM_TENSORS] =
    [true, true, false, false, true, false, true, false, false, false, true, false, true, false, false, false];

/// All learnable weights. Per-layer tensors are stacked along the first axis.
/// The same struct also holds gradients and the optimizer's moment estimates.
#[derive(Clone)]
pub struct Params {
    pub wte: Vec<f32>,      // [V, C]  token embeddings (tied with the output head)
    pub wpe: Vec<f32>,      // [T, C]  position embeddings
    pub ln1w: Vec<f32>,     // [L, C]
    pub ln1b: Vec<f32>,     // [L, C]
    pub qkvw: Vec<f32>,     // [L, 3C, C]
    pub qkvb: Vec<f32>,     // [L, 3C]
    pub attprojw: Vec<f32>, // [L, C, C]
    pub attprojb: Vec<f32>, // [L, C]
    pub ln2w: Vec<f32>,     // [L, C]
    pub ln2b: Vec<f32>,     // [L, C]
    pub fcw: Vec<f32>,      // [L, 4C, C]
    pub fcb: Vec<f32>,      // [L, 4C]
    pub fcprojw: Vec<f32>,  // [L, C, 4C]
    pub fcprojb: Vec<f32>,  // [L, C]
    pub lnfw: Vec<f32>,     // [C]
    pub lnfb: Vec<f32>,     // [C]
}

impl Params {
    pub fn sizes(cfg: &Config) -> [usize; NUM_TENSORS] {
        let (v, t, l, c) = (cfg.vocab_size, cfg.block_size, cfg.n_layer, cfg.n_embd);
        [
            v * c,
            t * c,
            l * c,
            l * c,
            l * 3 * c * c,
            l * 3 * c,
            l * c * c,
            l * c,
            l * c,
            l * c,
            l * 4 * c * c,
            l * 4 * c,
            l * c * 4 * c,
            l * c,
            c,
            c,
        ]
    }

    pub fn zeros(cfg: &Config) -> Self {
        let [wte, wpe, ln1w, ln1b, qkvw, qkvb, attprojw, attprojb, ln2w, ln2b, fcw, fcb, fcprojw, fcprojb, lnfw, lnfb] =
            Self::sizes(cfg).map(|n| vec![0.0; n]);
        Params {
            wte,
            wpe,
            ln1w,
            ln1b,
            qkvw,
            qkvb,
            attprojw,
            attprojb,
            ln2w,
            ln2b,
            fcw,
            fcb,
            fcprojw,
            fcprojb,
            lnfw,
            lnfb,
        }
    }

    pub fn tensors(&self) -> [&Vec<f32>; NUM_TENSORS] {
        [
            &self.wte,
            &self.wpe,
            &self.ln1w,
            &self.ln1b,
            &self.qkvw,
            &self.qkvb,
            &self.attprojw,
            &self.attprojb,
            &self.ln2w,
            &self.ln2b,
            &self.fcw,
            &self.fcb,
            &self.fcprojw,
            &self.fcprojb,
            &self.lnfw,
            &self.lnfb,
        ]
    }

    pub fn tensors_mut(&mut self) -> [&mut Vec<f32>; NUM_TENSORS] {
        [
            &mut self.wte,
            &mut self.wpe,
            &mut self.ln1w,
            &mut self.ln1b,
            &mut self.qkvw,
            &mut self.qkvb,
            &mut self.attprojw,
            &mut self.attprojb,
            &mut self.ln2w,
            &mut self.ln2b,
            &mut self.fcw,
            &mut self.fcb,
            &mut self.fcprojw,
            &mut self.fcprojb,
            &mut self.lnfw,
            &mut self.lnfb,
        ]
    }

    pub fn num_params(&self) -> usize {
        self.tensors().iter().map(|t| t.len()).sum()
    }

    pub fn zero(&mut self) {
        self.tensors_mut().into_iter().for_each(|t| t.fill(0.0));
    }

    pub fn norm(&self) -> f32 {
        let sq: f64 = self.tensors().iter().flat_map(|t| t.iter()).map(|&x| (x as f64) * (x as f64)).sum();
        sq.sqrt() as f32
    }

    pub fn scale(&mut self, s: f32) {
        self.tensors_mut().into_iter().flatten().for_each(|x| *x *= s);
    }
}

fn ls(v: &[f32], l: usize, n: usize) -> &[f32] {
    &v[l * n..(l + 1) * n]
}

fn lsm(v: &mut [f32], l: usize, n: usize) -> &mut [f32] {
    &mut v[l * n..(l + 1) * n]
}

/// Everything the forward pass computes that the backward pass needs, plus
/// scratch buffers for activation gradients. Sized for one (B, T) shape.
struct Acts {
    b: usize,
    t: usize,
    encoded: Vec<f32>,   // [B, T, C]
    ln1: Vec<f32>,       // [L, B, T, C]
    ln1_mean: Vec<f32>,  // [L, B, T]
    ln1_rstd: Vec<f32>,  // [L, B, T]
    qkv: Vec<f32>,       // [L, B, T, 3C]
    atty: Vec<f32>,      // [L, B, T, C]
    att: Vec<f32>,       // [L, B, NH, T, T]
    attproj: Vec<f32>,   // [L, B, T, C]
    residual2: Vec<f32>, // [L, B, T, C]
    ln2: Vec<f32>,       // [L, B, T, C]
    ln2_mean: Vec<f32>,  // [L, B, T]
    ln2_rstd: Vec<f32>,  // [L, B, T]
    fch: Vec<f32>,       // [L, B, T, 4C]
    fch_gelu: Vec<f32>,  // [L, B, T, 4C]
    fcproj: Vec<f32>,    // [L, B, T, C]
    residual3: Vec<f32>, // [L, B, T, C]
    lnf: Vec<f32>,       // [B, T, C]
    lnf_mean: Vec<f32>,  // [B, T]
    lnf_rstd: Vec<f32>,  // [B, T]
    logits: Vec<f32>,    // [B, T, V]
    probs: Vec<f32>,     // [B, T, V]
    // Gradient scratch, reused across layers.
    dlogits: Vec<f32>,   // [B, T, V]
    dlnf: Vec<f32>,      // [B, T, C]
    dresidual: Vec<f32>, // [B, T, C]
    dln: Vec<f32>,       // [B, T, C]
    datty: Vec<f32>,     // [B, T, C]
    dqkv: Vec<f32>,      // [B, T, 3C]
    dfch: Vec<f32>,      // [B, T, 4C]
    dfch_gelu: Vec<f32>, // [B, T, 4C]
}

impl Acts {
    fn new(cfg: &Config, b: usize, t: usize) -> Self {
        let (l, c, v, nh) = (cfg.n_layer, cfg.n_embd, cfg.vocab_size, cfg.n_head);
        let n = b * t;
        let z = |k: usize| vec![0.0f32; k];
        Acts {
            b,
            t,
            encoded: z(n * c),
            ln1: z(l * n * c),
            ln1_mean: z(l * n),
            ln1_rstd: z(l * n),
            qkv: z(l * n * 3 * c),
            atty: z(l * n * c),
            att: z(l * b * nh * t * t),
            attproj: z(l * n * c),
            residual2: z(l * n * c),
            ln2: z(l * n * c),
            ln2_mean: z(l * n),
            ln2_rstd: z(l * n),
            fch: z(l * n * 4 * c),
            fch_gelu: z(l * n * 4 * c),
            fcproj: z(l * n * c),
            residual3: z(l * n * c),
            lnf: z(n * c),
            lnf_mean: z(n),
            lnf_rstd: z(n),
            logits: z(n * v),
            probs: z(n * v),
            dlogits: z(n * v),
            dlnf: z(n * c),
            dresidual: z(n * c),
            dln: z(n * c),
            datty: z(n * c),
            dqkv: z(n * 3 * c),
            dfch: z(n * 4 * c),
            dfch_gelu: z(n * 4 * c),
        }
    }
}

pub struct Gpt {
    pub cfg: Config,
    pub params: Params,
    pub grads: Params,
    acts: Option<Acts>,
    inputs: Vec<usize>,
    targets: Vec<usize>,
}

impl Gpt {
    pub fn new(cfg: Config, rng: &mut Rng) -> Self {
        assert!(cfg.n_embd.is_multiple_of(cfg.n_head), "n_embd must be divisible by n_head");
        let mut params = Params::zeros(&cfg);
        // GPT-2 init: N(0, 0.02), residual projections scaled down by sqrt(2 * n_layer).
        let resid_std = 0.02 / (2.0 * cfg.n_layer as f32).sqrt();
        for (i, t) in params.tensors_mut().into_iter().enumerate() {
            match PARAM_NAMES[i] {
                "ln1w" | "ln2w" | "lnfw" => t.fill(1.0),
                "attprojw" | "fcprojw" => t.iter_mut().for_each(|x| *x = resid_std * rng.normal()),
                _ if DECAY[i] => t.iter_mut().for_each(|x| *x = 0.02 * rng.normal()),
                _ => {} // biases start at zero
            }
        }
        Self::from_params(cfg, params)
    }

    pub fn from_params(cfg: Config, params: Params) -> Self {
        Gpt { cfg, grads: Params::zeros(&cfg), params, acts: None, inputs: vec![], targets: vec![] }
    }

    /// Runs the model on `inputs` (`[b, t]` token ids). With `targets`, also
    /// computes and returns the mean cross-entropy loss.
    pub fn forward(&mut self, inputs: &[usize], targets: Option<&[usize]>, b: usize, t: usize) -> Option<f32> {
        let cfg = self.cfg;
        assert!(t >= 1 && t <= cfg.block_size, "sequence length {t} exceeds block size");
        assert_eq!(inputs.len(), b * t);
        let (v, nl, nh, c) = (cfg.vocab_size, cfg.n_layer, cfg.n_head, cfg.n_embd);
        if self.acts.as_ref().is_none_or(|a| a.b != b || a.t != t) {
            self.acts = Some(Acts::new(&cfg, b, t));
        }
        let p = &self.params;
        let a = self.acts.as_mut().unwrap();
        let n = b * t;
        let btc = n * c;
        let att_sz = b * nh * t * t;

        encoder_forward(&mut a.encoded, inputs, &p.wte, &p.wpe, t, c);
        for l in 0..nl {
            let (res_prev, res_cur) = a.residual3.split_at_mut(l * btc);
            let residual: &[f32] = if l == 0 { &a.encoded } else { ls(res_prev, l - 1, btc) };

            // x = x + attn(ln1(x))
            let ln1 = lsm(&mut a.ln1, l, btc);
            layernorm_forward(
                ln1,
                lsm(&mut a.ln1_mean, l, n),
                lsm(&mut a.ln1_rstd, l, n),
                residual,
                ls(&p.ln1w, l, c),
                ls(&p.ln1b, l, c),
                c,
            );
            let qkv = lsm(&mut a.qkv, l, 3 * btc);
            matmul_forward(qkv, ln1, ls(&p.qkvw, l, 3 * c * c), Some(ls(&p.qkvb, l, 3 * c)), c, 3 * c);
            let atty = lsm(&mut a.atty, l, btc);
            attention_forward(atty, lsm(&mut a.att, l, att_sz), qkv, t, c, nh);
            let attproj = lsm(&mut a.attproj, l, btc);
            matmul_forward(attproj, atty, ls(&p.attprojw, l, c * c), Some(ls(&p.attprojb, l, c)), c, c);
            let res2 = lsm(&mut a.residual2, l, btc);
            residual_forward(res2, residual, attproj);

            // x = x + mlp(ln2(x))
            let ln2 = lsm(&mut a.ln2, l, btc);
            layernorm_forward(
                ln2,
                lsm(&mut a.ln2_mean, l, n),
                lsm(&mut a.ln2_rstd, l, n),
                res2,
                ls(&p.ln2w, l, c),
                ls(&p.ln2b, l, c),
                c,
            );
            let fch = lsm(&mut a.fch, l, 4 * btc);
            matmul_forward(fch, ln2, ls(&p.fcw, l, 4 * c * c), Some(ls(&p.fcb, l, 4 * c)), c, 4 * c);
            let fch_gelu = lsm(&mut a.fch_gelu, l, 4 * btc);
            gelu_forward(fch_gelu, fch);
            let fcproj = lsm(&mut a.fcproj, l, btc);
            matmul_forward(fcproj, fch_gelu, ls(&p.fcprojw, l, 4 * c * c), Some(ls(&p.fcprojb, l, c)), 4 * c, c);
            residual_forward(&mut res_cur[..btc], res2, fcproj);
        }

        let last = ls(&a.residual3, nl - 1, btc);
        layernorm_forward(&mut a.lnf, &mut a.lnf_mean, &mut a.lnf_rstd, last, &p.lnfw, &p.lnfb, c);
        matmul_forward(&mut a.logits, &a.lnf, &p.wte, None, c, v);

        self.inputs.clear();
        self.inputs.extend_from_slice(inputs);
        let targets = targets?;
        assert_eq!(targets.len(), n);
        let mut loss = 0.0f64;
        for ((prow, lrow), &tg) in a.probs.chunks_mut(v).zip(a.logits.chunks(v)).zip(targets) {
            softmax(prow, lrow);
            loss -= (prow[tg].max(1e-30) as f64).ln();
        }
        self.targets.clear();
        self.targets.extend_from_slice(targets);
        Some((loss / n as f64) as f32)
    }

    /// Backpropagates the loss from the most recent `forward` call with targets
    /// into `self.grads` (which is overwritten).
    pub fn backward(&mut self) {
        let cfg = self.cfg;
        let (v, nl, nh, c) = (cfg.vocab_size, cfg.n_layer, cfg.n_head, cfg.n_embd);
        let a = self.acts.as_mut().expect("call forward before backward");
        assert_eq!(self.targets.len(), a.b * a.t, "backward needs a forward pass with targets");
        let p = &self.params;
        let g = &mut self.grads;
        g.zero();
        let (b, t) = (a.b, a.t);
        let n = b * t;
        let btc = n * c;
        let att_sz = b * nh * t * t;

        // d(mean cross-entropy)/d(logits) = (softmax - onehot) / N
        let scale = 1.0 / n as f32;
        for ((drow, prow), &tg) in a.dlogits.chunks_mut(v).zip(a.probs.chunks(v)).zip(&self.targets) {
            for (d, &pr) in drow.iter_mut().zip(prow) {
                *d = pr * scale;
            }
            drow[tg] -= scale;
        }
        matmul_backward(&mut a.dlnf, &mut g.wte, None, &a.dlogits, &a.lnf, &p.wte, c, v);
        a.dresidual.fill(0.0);
        layernorm_backward(
            &mut a.dresidual,
            &mut g.lnfw,
            &mut g.lnfb,
            &a.dlnf,
            ls(&a.residual3, nl - 1, btc),
            &p.lnfw,
            &a.lnf_mean,
            &a.lnf_rstd,
            c,
        );

        // `dresidual` flows straight down the residual stream; each block adds its contribution.
        for l in (0..nl).rev() {
            let residual: &[f32] = if l == 0 { &a.encoded } else { ls(&a.residual3, l - 1, btc) };

            matmul_backward(
                &mut a.dfch_gelu,
                lsm(&mut g.fcprojw, l, 4 * c * c),
                Some(lsm(&mut g.fcprojb, l, c)),
                &a.dresidual,
                ls(&a.fch_gelu, l, 4 * btc),
                ls(&p.fcprojw, l, 4 * c * c),
                4 * c,
                c,
            );
            gelu_backward(&mut a.dfch, ls(&a.fch, l, 4 * btc), &a.dfch_gelu);
            matmul_backward(
                &mut a.dln,
                lsm(&mut g.fcw, l, 4 * c * c),
                Some(lsm(&mut g.fcb, l, 4 * c)),
                &a.dfch,
                ls(&a.ln2, l, btc),
                ls(&p.fcw, l, 4 * c * c),
                c,
                4 * c,
            );
            layernorm_backward(
                &mut a.dresidual,
                lsm(&mut g.ln2w, l, c),
                lsm(&mut g.ln2b, l, c),
                &a.dln,
                ls(&a.residual2, l, btc),
                ls(&p.ln2w, l, c),
                ls(&a.ln2_mean, l, n),
                ls(&a.ln2_rstd, l, n),
                c,
            );

            matmul_backward(
                &mut a.datty,
                lsm(&mut g.attprojw, l, c * c),
                Some(lsm(&mut g.attprojb, l, c)),
                &a.dresidual,
                ls(&a.atty, l, btc),
                ls(&p.attprojw, l, c * c),
                c,
                c,
            );
            attention_backward(&mut a.dqkv, &a.datty, ls(&a.qkv, l, 3 * btc), ls(&a.att, l, att_sz), t, c, nh);
            matmul_backward(
                &mut a.dln,
                lsm(&mut g.qkvw, l, 3 * c * c),
                Some(lsm(&mut g.qkvb, l, 3 * c)),
                &a.dqkv,
                ls(&a.ln1, l, btc),
                ls(&p.qkvw, l, 3 * c * c),
                c,
                3 * c,
            );
            layernorm_backward(
                &mut a.dresidual,
                lsm(&mut g.ln1w, l, c),
                lsm(&mut g.ln1b, l, c),
                &a.dln,
                residual,
                ls(&p.ln1w, l, c),
                ls(&a.ln1_mean, l, n),
                ls(&a.ln1_rstd, l, n),
                c,
            );
        }
        encoder_backward(&mut g.wte, &mut g.wpe, &a.dresidual, &self.inputs, t, c);
    }

    /// Logits for position `pos` of batch row 0 from the most recent forward pass.
    pub fn logits_at(&self, pos: usize) -> &[f32] {
        let a = self.acts.as_ref().expect("call forward first");
        let v = self.cfg.vocab_size;
        &a.logits[pos * v..(pos + 1) * v]
    }

    /// Attention weights `[T, T]` of one layer/head for batch row 0 from the most recent forward pass.
    pub fn attention_map(&self, layer: usize, head: usize) -> &[f32] {
        let a = self.acts.as_ref().expect("call forward first");
        let (t, nh) = (a.t, self.cfg.n_head);
        let per_layer = a.b * nh * t * t;
        &a.att[layer * per_layer + head * t * t..][..t * t]
    }

    /// Samples the next token given a context, using temperature and top-k.
    pub fn sample_next(&mut self, context: &[usize], temperature: f32, top_k: usize, rng: &mut Rng) -> usize {
        let start = context.len().saturating_sub(self.cfg.block_size);
        let ctx = &context[start..];
        self.forward(ctx, None, 1, ctx.len());
        let logits = self.logits_at(ctx.len() - 1);
        sample(logits, temperature, top_k, rng)
    }
}

pub fn sample(logits: &[f32], temperature: f32, top_k: usize, rng: &mut Rng) -> usize {
    if temperature <= 0.0 {
        return argmax(logits);
    }
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
    if top_k > 0 {
        idx.truncate(top_k);
    }
    let scaled: Vec<f32> = idx.iter().map(|&i| logits[i] / temperature).collect();
    let mut probs = vec![0.0; scaled.len()];
    softmax(&mut probs, &scaled);
    let r = rng.uniform();
    let mut cum = 0.0;
    for (k, &pr) in probs.iter().enumerate() {
        cum += pr;
        if r < cum {
            return idx[k];
        }
    }
    idx[probs.len() - 1]
}

fn argmax(x: &[f32]) -> usize {
    (0..x.len()).max_by(|&a, &b| x[a].total_cmp(&x[b])).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny() -> (Gpt, Vec<usize>, Vec<usize>, usize, usize) {
        let cfg = Config { vocab_size: 11, block_size: 8, n_layer: 2, n_head: 2, n_embd: 16 };
        let mut rng = Rng::new(42);
        let mut model = Gpt::new(cfg, &mut rng);
        // Larger weights than the real init so every path carries a meaningful gradient.
        for t in model.params.tensors_mut() {
            t.iter_mut().for_each(|x| *x += 0.2 * rng.normal());
        }
        let (b, t) = (2, 8);
        let x: Vec<usize> = (0..b * t).map(|_| rng.below(11)).collect();
        let y: Vec<usize> = (0..b * t).map(|_| rng.below(11)).collect();
        (model, x, y, b, t)
    }

    /// Compares every hand-written backward pass against finite differences.
    #[test]
    #[allow(clippy::needless_range_loop)]
    fn gradients_match_finite_differences() {
        let (mut model, x, y, b, t) = tiny();
        model.forward(&x, Some(&y), b, t);
        model.backward();
        let grads = model.grads.clone();
        let mut rng = Rng::new(7);
        let eps = 1e-2f32;
        for ti in 0..NUM_TENSORS {
            let len = model.params.tensors()[ti].len();
            for _ in 0..6 {
                let i = rng.below(len);
                let orig = model.params.tensors()[ti][i];
                model.params.tensors_mut()[ti][i] = orig + eps;
                let lp = model.forward(&x, Some(&y), b, t).unwrap() as f64;
                model.params.tensors_mut()[ti][i] = orig - eps;
                let lm = model.forward(&x, Some(&y), b, t).unwrap() as f64;
                model.params.tensors_mut()[ti][i] = orig;
                let numeric = ((lp - lm) / (2.0 * eps as f64)) as f32;
                let analytic = grads.tensors()[ti][i];
                let tol = 2e-3 + 0.05 * analytic.abs().max(numeric.abs());
                assert!(
                    (numeric - analytic).abs() < tol,
                    "{}[{i}]: analytic {analytic} vs numeric {numeric}",
                    PARAM_NAMES[ti]
                );
            }
        }
    }

    #[test]
    fn attention_is_causal() {
        let (mut model, mut x, _, b, t) = tiny();
        model.forward(&x, None, b, t);
        let before: Vec<f32> = model.logits_at(3).to_vec();
        for tok in &mut x[4..t] {
            *tok = (*tok + 1) % 11; // change only the future
        }
        model.forward(&x, None, b, t);
        assert_eq!(before, model.logits_at(3), "position 3 must not see positions 4+");
    }

    #[test]
    fn can_overfit_a_batch() {
        let (mut model, x, y, b, t) = tiny();
        let mut opt = crate::optim::AdamW::new(&model.cfg, 0.0);
        let first = model.forward(&x, Some(&y), b, t).unwrap();
        let mut last = first;
        for _ in 0..150 {
            last = model.forward(&x, Some(&y), b, t).unwrap();
            model.backward();
            opt.update(&mut model.params, &model.grads, 3e-3);
        }
        assert!(last < 0.1 * first, "loss went {first} -> {last}");
    }
}
