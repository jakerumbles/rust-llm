use std::fs;
use anyhow::{Result, Context};

use crate::tokenizer::Tokenizer;

pub struct Dataset {
    pub tokenizer: Tokenizer,
    pub train: Vec<u32>,
    pub val: Vec<u32>
}

impl Dataset {
    /// Reads the corpus of data at the file `path`, then creates the tokenizer from it, encodes it, and 
    /// splits the data 90% / 10% for training and test sets.
    pub fn load(path: &str) -> Result<Self> {
        let corpus = fs::read_to_string(path).context("Failed to load corpus data")?;

        // Create our tokenizer
        let tokenizer = Tokenizer::new(&corpus);
        let mut encoded_text = tokenizer.encode(&corpus);

        // 90% / 10% training data split
        let split_idx = encoded_text.len() * 9 / 10;

        let val_data = encoded_text.split_off(split_idx);
        let train_data = encoded_text;

        Ok(Dataset{
            tokenizer,
            train: train_data,
            val: val_data
        })
    }
}
