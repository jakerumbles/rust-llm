use std::collections::{HashMap, BTreeSet};

pub struct Tokenizer {
    char_to_id: HashMap<char, u32>,
    id_to_char: HashMap<u32, char>,
    vocab_size: usize,
}

impl Tokenizer {
    /// Create a new tokenizer on a corpus of text
    pub fn new(corpus: &str) -> Self {
        let mut char_to_id: HashMap<char, u32> = HashMap::new();
        let mut id_to_char: HashMap<u32, char> = HashMap::new();

        // Get set of chars in corpus
        let unique_chars: BTreeSet<char> = corpus.chars().collect();
        // println!("unique chars: {unique_chars:?}");

        let vocab_size = unique_chars.len();

        // Seed tokenizers
        for (index, token) in unique_chars.into_iter().enumerate() {
            char_to_id.insert(token, index as u32);
            id_to_char.insert(index as u32, token);
        }

        // Debugging
        // for (key, value) in &id_to_char {
        //     println!("{key}: {value}");
        // }

        Tokenizer { char_to_id, id_to_char, vocab_size }
    }

    /// Using the encoding mapping generated when the `new` fn was called,
    /// encode the given text slice and return it's encoded as u32's
    pub fn encode(&self, text: &str) -> Vec<u32> {    
        // Encode the text
        let encoded_text: Vec<u32> = text.chars().map(|letter| {
            self.char_to_id.get(&letter).unwrap().clone()
        }).collect();

        encoded_text
    }
    
    /// Using the encoding mapping generated when the `new` fn was called,
    /// decode the given encoded data and return the decoded data as text
    pub fn decode(&self, tokens: &[u32]) -> String {
        let decoded_characters: String = tokens.iter().map(|token| {
            self.id_to_char.get(token).unwrap()
        }).collect();

        decoded_characters
    }
}



