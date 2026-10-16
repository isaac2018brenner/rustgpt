//! A small self-describing binary checkpoint: config, vocabulary, then raw f32 weights.

use crate::data::Tokenizer;
use crate::model::{Config, Params};
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

const MAGIC: &[u8; 4] = b"RGPT";
const VERSION: u32 = 1;

pub fn save(path: &Path, cfg: &Config, tok: &Tokenizer, params: &Params) -> io::Result<()> {
    let mut w = BufWriter::new(File::create(path)?);
    w.write_all(MAGIC)?;
    for x in [VERSION as usize, cfg.vocab_size, cfg.block_size, cfg.n_layer, cfg.n_head, cfg.n_embd] {
        w.write_all(&(x as u32).to_le_bytes())?;
    }
    let vocab: String = tok.chars().iter().collect();
    w.write_all(&(vocab.len() as u32).to_le_bytes())?;
    w.write_all(vocab.as_bytes())?;
    for t in params.tensors() {
        for x in t.iter() {
            w.write_all(&x.to_le_bytes())?;
        }
    }
    w.flush()
}

pub fn load(path: &Path) -> io::Result<(Config, Tokenizer, Params)> {
    let mut r = BufReader::new(File::open(path)?);
    let bad = |msg: &str| io::Error::new(io::ErrorKind::InvalidData, msg.to_string());
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(bad("not a rustgpt checkpoint"));
    }
    let mut u32s = [0usize; 7];
    for x in &mut u32s {
        let mut buf = [0u8; 4];
        r.read_exact(&mut buf)?;
        *x = u32::from_le_bytes(buf) as usize;
    }
    let [version, vocab_size, block_size, n_layer, n_head, n_embd, vocab_len] = u32s;
    if version != VERSION as usize {
        return Err(bad("unsupported checkpoint version"));
    }
    let cfg = Config { vocab_size, block_size, n_layer, n_head, n_embd };
    let mut vocab = vec![0u8; vocab_len];
    r.read_exact(&mut vocab)?;
    let vocab = String::from_utf8(vocab).map_err(|_| bad("vocabulary is not utf-8"))?;
    let tok = Tokenizer::from_chars(vocab.chars().collect());
    if tok.vocab_size() != vocab_size {
        return Err(bad("vocabulary size mismatch"));
    }
    let mut params = Params::zeros(&cfg);
    for t in params.tensors_mut() {
        let mut buf = vec![0u8; t.len() * 4];
        r.read_exact(&mut buf)?;
        for (x, b) in t.iter_mut().zip(buf.chunks_exact(4)) {
            *x = f32::from_le_bytes(b.try_into().unwrap());
        }
    }
    Ok((cfg, tok, params))
}
