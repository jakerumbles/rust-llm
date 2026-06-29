mod data;
mod tokenizer;
mod batch;

use data::Dataset;
use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use batch::get_batch;

const BATCH_SIZE: usize = 3;
const BLOCK_SIZE: usize = 4;

fn main() -> Result<()> {
    let dataset = Dataset::load("data/input.txt").context("Failed to setup dataset")?;

    let device = Device::Cpu;

    // let t = Tensor::new(&[1u32, 2, 3, 4, 5, 6, 7, 8], &device)?.reshape((2, 4))?;
    // println!("Shape: {:?}, dims: {:?}, type: {:?}", t.shape(), t.dims(), t.dtype());
    // println!("{:?}", t);

    let (xb, yb) = get_batch(&dataset.train, BATCH_SIZE, BLOCK_SIZE, &device).context("Failed to get batches")?;

    Ok(())
}
