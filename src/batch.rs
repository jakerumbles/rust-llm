use candle_core::{Device, Tensor};
use anyhow::Context;

/// Create our batch tensors given `data` which is a reference to the entire training data vec
pub fn get_batch(data: &[u32], batch_size: usize, block_size: usize, device: &Device) -> anyhow::Result<(Tensor, Tensor)> {
    // Batch Sampling
    let train_length = data.len() - block_size - 1;

    let mut x_buf: Vec<u32> = Vec::with_capacity(batch_size * block_size);
    let mut y_buf: Vec<u32> = Vec::with_capacity(batch_size * block_size);

    // Generate batches of input and output
    for _ in 0..batch_size {
        // Generate random training set index for batch (tensor) row
        let rand_index = rand::random_range(0..train_length);

        // Contains all `block_size` + 1 tokens which is contains token windows for x and y
        let window = &data[rand_index..rand_index + block_size + 1];
        x_buf.extend_from_slice(&window[..block_size]);
        y_buf.extend_from_slice(&window[1..]);
    }

    // Build tensors for input and output
    let x = Tensor::from_vec(x_buf, (batch_size, block_size), &device).context("Error creating x tensor batch")?;
    let y = Tensor::from_vec(y_buf, (batch_size, block_size), &device).context("Error creating y tensor batch")?;

    // Debug
    // println!("x: {}", xb);
    // println!("y: {}", yb);

    // for row in xb.to_vec2::<u32>()? {
    //     println!("x Row: {}", dataset.tokenizer.decode(&row));
    // }

    // for row in yb.to_vec2::<u32>()? {
    //     println!("y Row: {}", dataset.tokenizer.decode(&row));
    // }

    Ok((x, y))
}