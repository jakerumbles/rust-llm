mod data;
mod tokenizer;
mod batch;

use data::Dataset;
use anyhow::{Context, Result};
use candle_core::{Device, Tensor, DType, Module};
use candle_nn::{embedding, VarMap, VarBuilder};
use batch::get_batches;

const BATCH_SIZE: usize = 4; // B
const BLOCK_SIZE: usize = 8; // T
const EMBEDDING_DIMENSION: usize = 32; // C

fn main() -> Result<()> {
    let dataset = Dataset::load("data/input.txt").context("Failed to setup dataset")?;

    let device = Device::Cpu;

    // let t = Tensor::new(&[1u32, 2, 3, 4, 5, 6, 7, 8], &device)?.reshape((2, 4))?;
    // println!("Shape: {:?}, dims: {:?}, type: {:?}", t.shape(), t.dims(), t.dtype());
    // println!("{:?}", t);

    // Build our training batches of input and output
    let (xb, yb) = get_batches(&dataset.train, BATCH_SIZE, BLOCK_SIZE, &device).context("Failed to get batches")?;

    println!("x batches: {xb}");

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    // Token embedding: a learned lookup table that turns each token ID into an EMBEDDING_DIMENSION vector. First trainable layer; pure per-token, no cross-token reasoning yet
    let token_emb = embedding(dataset.tokenizer.vocab_size, EMBEDDING_DIMENSION, vb.pp("token_emb"))?;
    let position_emb = embedding(BLOCK_SIZE, EMBEDDING_DIMENSION, vb.pp("position_emb"))?;

    let tok = token_emb.forward(&xb)?;
    println!("Forward pass returned tensor: {}", tok);

    let position = Tensor::arange(0u32, BLOCK_SIZE as u32, &device)?;
    let pos = position_emb.forward(&position)?;
    let x = tok.broadcast_add(&pos)?;

    Ok(())
}
