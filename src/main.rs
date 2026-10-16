use rustgpt::checkpoint;
use rustgpt::data::{get_batch, Tokenizer};
use rustgpt::model::{Config, Gpt};
use rustgpt::ops::softmax;
use rustgpt::optim::AdamW;
use rustgpt::parallel::num_threads;
use rustgpt::rng::Rng;
use std::collections::HashMap;
use std::io::{IsTerminal, Write};
use std::path::Path;
use std::process::exit;
use std::time::Instant;

const USAGE: &str = "\
rustgpt: a GPT built from scratch in pure Rust

USAGE:
  rustgpt train    [--data data/input.txt] [--out model.bin] [--steps 3000]
                   [--batch 16] [--block 64] [--layers 4] [--heads 4] [--embd 128]
                   [--lr 1e-3] [--warmup 100] [--wd 0.1] [--clip 1.0]
                   [--eval-every 250] [--eval-iters 20] [--seed 1337]
  rustgpt generate [--model model.bin] [--prompt TEXT] [--tokens 500]
                   [--temperature 0.8] [--top-k 0] [--seed 42]
  rustgpt attn     [--model model.bin] --text TEXT [--layer N]
  rustgpt info     [--model model.bin]

Set RUSTGPT_THREADS to control parallelism.";

struct Args(HashMap<String, String>);

impl Args {
    fn get<T: std::str::FromStr>(&self, key: &str, default: T) -> T {
        match self.0.get(key) {
            None => default,
            Some(v) => v.parse().unwrap_or_else(|_| fail(&format!("invalid value for --{key}: {v}"))),
        }
    }

    fn str(&self, key: &str, default: &str) -> String {
        self.0.get(key).cloned().unwrap_or_else(|| default.to_string())
    }
}

fn fail(msg: &str) -> ! {
    eprintln!("error: {msg}\n\n{USAGE}");
    exit(1);
}

fn parse_args() -> (String, Args) {
    let mut it = std::env::args().skip(1);
    let cmd = it.next().unwrap_or_else(|| fail("missing command"));
    let mut map = HashMap::new();
    while let Some(k) = it.next() {
        let key = k.strip_prefix("--").unwrap_or_else(|| fail(&format!("unexpected argument {k}")));
        let val = it.next().unwrap_or_else(|| fail(&format!("--{key} needs a value")));
        map.insert(key.to_string(), val);
    }
    (cmd, Args(map))
}

fn main() {
    let (cmd, args) = parse_args();
    match cmd.as_str() {
        "train" => train(&args),
        "generate" => generate(&args),
        "attn" => attn(&args),
        "info" => info(&args),
        "help" | "--help" | "-h" => println!("{USAGE}"),
        _ => fail(&format!("unknown command {cmd}")),
    }
}

fn load_model(args: &Args) -> (Gpt, Tokenizer) {
    let path = args.str("model", "model.bin");
    let (cfg, tok, params) = checkpoint::load(Path::new(&path))
        .unwrap_or_else(|e| fail(&format!("could not load {path}: {e} (run `rustgpt train` first)")));
    (Gpt::from_params(cfg, params), tok)
}

fn train(args: &Args) {
    let data_path = args.str("data", "data/input.txt");
    let out = args.str("out", "model.bin");
    let steps: usize = args.get("steps", 3000);
    let b: usize = args.get("batch", 16);
    let t: usize = args.get("block", 64);
    let max_lr: f32 = args.get("lr", 1e-3);
    let warmup: usize = args.get("warmup", 100);
    let clip: f32 = args.get("clip", 1.0);
    let eval_every: usize = args.get("eval-every", 250);
    let eval_iters: usize = args.get("eval-iters", 20);
    let mut rng = Rng::new(args.get("seed", 1337));

    let text =
        std::fs::read_to_string(&data_path).unwrap_or_else(|e| fail(&format!("could not read {data_path}: {e}")));
    let tok = Tokenizer::from_text(&text);
    let tokens = tok.encode(&text);
    let split = tokens.len() * 9 / 10;
    let (train_data, val_data) = tokens.split_at(split);

    let cfg = Config {
        vocab_size: tok.vocab_size(),
        block_size: t,
        n_layer: args.get("layers", 4),
        n_head: args.get("heads", 4),
        n_embd: args.get("embd", 128),
    };
    let mut model = Gpt::new(cfg, &mut rng);
    let mut opt = AdamW::new(&cfg, args.get("wd", 0.1));

    println!("dataset   {data_path}: {} chars, vocab {}", tokens.len(), cfg.vocab_size);
    println!(
        "model     {} layers, {} heads, {} dim, context {} -> {:.2}M parameters",
        cfg.n_layer,
        cfg.n_head,
        cfg.n_embd,
        cfg.block_size,
        model.params.num_params() as f64 / 1e6
    );
    println!("training  {steps} steps, batch {b}x{t}, lr {max_lr}, {} threads\n", num_threads());

    let mut best_val = f32::INFINITY;
    let started = Instant::now();
    let mut window = Instant::now();
    for step in 1..=steps {
        // Linear warmup, then cosine decay to 10% of the peak learning rate.
        let lr = if step <= warmup {
            max_lr * step as f32 / warmup as f32
        } else {
            let progress = (step - warmup) as f32 / (steps - warmup).max(1) as f32;
            max_lr * (0.1 + 0.45 * (1.0 + (std::f32::consts::PI * progress).cos()))
        };

        let (x, y) = get_batch(train_data, b, t, &mut rng);
        let loss = model.forward(&x, Some(&y), b, t).unwrap();
        model.backward();
        let norm = model.grads.norm();
        if norm > clip {
            model.grads.scale(clip / norm);
        }
        opt.update(&mut model.params, &model.grads, lr);

        if step % 10 == 0 || step == 1 {
            let n = if step == 1 { 1 } else { 10 };
            let dt = window.elapsed().as_secs_f32();
            window = Instant::now();
            println!(
                "step {step:5}/{steps} | loss {loss:.4} | lr {lr:.2e} | grad norm {norm:6.3} | {:5.0} ms/step | {:6.0} tok/s",
                dt * 1000.0 / n as f32,
                (n * b * t) as f32 / dt
            );
        }

        if step % eval_every == 0 || step == steps {
            let mut eval_rng = Rng::new(0);
            let val: f32 = (0..eval_iters)
                .map(|_| {
                    let (x, y) = get_batch(val_data, b, t, &mut eval_rng);
                    model.forward(&x, Some(&y), b, t).unwrap()
                })
                .sum::<f32>()
                / eval_iters as f32;
            let mark = if val < best_val { "  (best, saved)" } else { "" };
            println!("\n=== step {step}: val loss {val:.4}{mark}");
            if val < best_val {
                best_val = val;
                checkpoint::save(Path::new(&out), &cfg, &tok, &model.params)
                    .unwrap_or_else(|e| fail(&format!("could not save {out}: {e}")));
            }
            let mut ctx = tok.encode("\n");
            for _ in 0..200 {
                let next = model.sample_next(&ctx, 0.8, 0, &mut rng);
                ctx.push(next);
            }
            println!("--- sample ---{}\n--------------\n", tok.decode(&ctx));
            window = Instant::now();
        }
    }
    println!(
        "done in {:.1} min. best val loss {best_val:.4}, checkpoint saved to {out}",
        started.elapsed().as_secs_f32() / 60.0
    );
}

fn generate(args: &Args) {
    let (mut model, tok) = load_model(args);
    let prompt = args.str("prompt", "\n");
    let n: usize = args.get("tokens", 500);
    let temperature: f32 = args.get("temperature", 0.8);
    let top_k: usize = args.get("top-k", 0);
    let mut rng = Rng::new(args.get("seed", 42));

    let mut ctx = tok.encode(&prompt);
    if ctx.is_empty() {
        fail("prompt has no characters from the model's vocabulary");
    }
    let mut out = std::io::stdout().lock();
    write!(out, "{}", tok.decode(&ctx)).unwrap();
    for _ in 0..n {
        let next = model.sample_next(&ctx, temperature, top_k, &mut rng);
        ctx.push(next);
        write!(out, "{}", tok.decode(&[next])).unwrap();
        out.flush().unwrap();
    }
    writeln!(out).unwrap();
}

fn info(args: &Args) {
    let (model, tok) = load_model(args);
    let cfg = model.cfg;
    println!("{cfg:#?}");
    println!("parameters: {}", model.params.num_params());
    println!("vocabulary: {:?}", tok.chars().iter().collect::<String>());
}

/// Shows what each attention head looks at, plus the model's next-character prediction.
fn attn(args: &Args) {
    let (mut model, tok) = load_model(args);
    let text = args.str("text", "");
    let mut ids = tok.encode(&text);
    if ids.is_empty() {
        fail("--text is required");
    }
    let t_max = model.cfg.block_size;
    if ids.len() > t_max {
        ids.drain(..ids.len() - t_max);
    }
    let t = ids.len();
    let layer: usize = args.get("layer", model.cfg.n_layer - 1);
    if layer >= model.cfg.n_layer {
        fail("--layer out of range");
    }
    model.forward(&ids, None, 1, t);

    let color = std::io::stdout().is_terminal();
    let ramp: Vec<char> = " .:-=+*#%@".chars().collect();
    let show = |c: char| match c {
        '\n' => '⏎',
        ' ' => '·',
        c => c,
    };
    let label: String = ids.iter().map(|&i| show(tok.chars()[i])).collect();
    println!("Attention in layer {layer} (rows: query position, columns: key position)\n");
    for h in 0..model.cfg.n_head {
        let att = model.attention_map(layer, h);
        println!("head {h}   {label}");
        for (q, row) in att.chunks(t).enumerate() {
            let mut line = String::new();
            for &w in &row[..=q] {
                let w = w.sqrt(); // sqrt makes weak-but-nonzero attention visible
                if color {
                    let shade = 232 + (w * 23.0).round() as u32;
                    line.push_str(&format!("\x1b[48;5;{shade}m \x1b[0m"));
                } else {
                    line.push(ramp[((w * (ramp.len() - 1) as f32).round() as usize).min(ramp.len() - 1)]);
                }
            }
            println!("     {} {line}", label.chars().nth(q).unwrap());
        }
        println!();
    }

    let mut probs = vec![0.0; model.cfg.vocab_size];
    softmax(&mut probs, model.logits_at(t - 1));
    let mut order: Vec<usize> = (0..probs.len()).collect();
    order.sort_by(|&a, &b| probs[b].total_cmp(&probs[a]));
    println!("Next character after {:?}:", text);
    for &i in order.iter().take(5) {
        let bar = "█".repeat((probs[i] * 40.0).round() as usize);
        println!("  {:>3?} {:5.1}% {bar}", tok.chars()[i], probs[i] * 100.0);
    }
}
