mod data;
mod tokenizer;

use data::get_corpus;
use tokenizer::Tokenizer;
use anyhow::{Context, Result};

fn main() -> Result<()> {
    let corpus = get_corpus("data/input.txt").context("Failed to get corpus")?;

    // Create our tokenizer
    let tokenizer = Tokenizer::new(&corpus);
    let encoded_text = tokenizer.encode(&corpus);
    let decoded_text = tokenizer.decode(&encoded_text);

    // println!("Encoded text: {encoded_text:?}");
    // println!("Decoded text: {decoded_text}");

    assert_eq!(corpus, decoded_text);

    Ok(())
}
