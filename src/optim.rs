//! AdamW with decoupled weight decay.

use crate::model::{Config, Params, DECAY};

pub struct AdamW {
    m: Params,
    v: Params,
    step: i32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
}

impl AdamW {
    pub fn new(cfg: &Config, weight_decay: f32) -> Self {
        AdamW {
            m: Params::zeros(cfg),
            v: Params::zeros(cfg),
            step: 0,
            beta1: 0.9,
            beta2: 0.95,
            eps: 1e-8,
            weight_decay,
        }
    }

    pub fn update(&mut self, params: &mut Params, grads: &Params, lr: f32) {
        self.step += 1;
        let bc1 = 1.0 - self.beta1.powi(self.step);
        let bc2 = 1.0 - self.beta2.powi(self.step);
        let (b1, b2, eps) = (self.beta1, self.beta2, self.eps);
        let tensors = params
            .tensors_mut()
            .into_iter()
            .zip(grads.tensors())
            .zip(self.m.tensors_mut())
            .zip(self.v.tensors_mut())
            .zip(DECAY);
        for ((((p, g), m), v), decay) in tensors {
            let wd = if decay { self.weight_decay } else { 0.0 };
            for (((p, &g), m), v) in p.iter_mut().zip(g.iter()).zip(m.iter_mut()).zip(v.iter_mut()) {
                *m = b1 * *m + (1.0 - b1) * g;
                *v = b2 * *v + (1.0 - b2) * g * g;
                let mhat = *m / bc1;
                let vhat = *v / bc2;
                *p -= lr * (mhat / (vhat.sqrt() + eps) + wd * *p);
            }
        }
    }
}
