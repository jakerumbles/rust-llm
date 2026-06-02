use std::fs;
use anyhow::Result;

pub fn get_corpus(filepath: &str) -> Result<String> {
    let corpus = fs::read_to_string(filepath)?;
    Ok(corpus)
}