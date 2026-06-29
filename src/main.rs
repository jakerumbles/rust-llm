mod data;
mod tokenizer;

use data::Dataset;
use tokenizer::Tokenizer;
use anyhow::{Context, Result};
use candle_core::{Device, Tensor};

fn main() -> Result<()> {
    let dataset = Dataset::load("data/input.txt").context("Failed to setup dataset")?;

    let device = Device::Cpu;

    let t = Tensor::new(&[1u32, 2, 3, 4, 5, 6, 7, 8], &device)?.reshape((2, 4))?;
    println!("Shape: {:?}, dims: {:?}, type: {:?}", t.shape(), t.dims(), t.dtype());
    println!("{:?}", t);
    Ok(())
}
