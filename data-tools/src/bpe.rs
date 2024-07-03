use std::{collections::HashMap, io::Write};

use base64::{prelude::BASE64_STANDARD, Engine};

type Token = u32;

const NUM_TOKENS_TO_GENERATE: usize = 2000;

pub struct Bpe {
    words: Vec<(Vec<Token>, u64)>,
    token_vocab: Vec<Vec<u8>>,
}

impl Bpe {
    pub fn new_and_run(words: Vec<(Vec<Token>, u64)>) -> Self {
        let mut result = Bpe {
            words,
            token_vocab: (0..=255).map(|x| [x].to_vec()).collect(),
        };
        result.run();
        result
    }

    fn find_most_common_pair(&self) -> Option<(Token, Token)> {
        let mut pairs = HashMap::new();
        for (word, word_count) in &self.words {
            let mut prev = word[0];
            for &token in &word[1..] {
                let pair = (prev, token);
                let count = pairs.entry(pair).or_insert(0);
                *count += word_count;
                prev = token;
            }
        }
        if pairs.is_empty() {
            None
        } else {
            Some(pairs.into_iter().max_by_key(|&(_, count)| count).unwrap().0)
        }
    }

    fn substitute_pair(&mut self, t0: Token, t1: Token, new_token: Token) {
        for (word, _) in &mut self.words {
            substitute_pair(word, t0, t1, new_token);
        }
    }

    fn step(&mut self) -> bool {
        if let Some((t0, t1)) = self.find_most_common_pair() {
            let new_token = self.token_vocab.len() as Token;
            self.substitute_pair(t0, t1, new_token);
            self.token_vocab.push(
                self.token_vocab[t0 as usize]
                    .iter()
                    .chain(&self.token_vocab[t1 as usize])
                    .cloned()
                    .collect(),
            );
            true
        } else {
            false
        }
    }

    pub fn run(&mut self) {
        while self.token_vocab.len() < NUM_TOKENS_TO_GENERATE {
            self.step();
            if self.token_vocab.len() % 100 == 0 {
                println!("Dictionary size: {}", self.token_vocab.len());
            }
        }
    }

    // pub fn into_dictionary(self) -> Vec<Vec<u8>> {
    //     self.token_vocab
    // }

    pub fn write_to_file(&self, filename: &str) {
        let mut file = std::fs::File::create(filename).unwrap();
        for token in &self.token_vocab {
            file.write_all(BASE64_STANDARD.encode(token).as_bytes()).unwrap();
            file.write_all(b"\n").unwrap();
        }
    }

    // pub fn get_token_counts(&self) -> Vec<u64> {
    //     let mut result = vec![0; self.token_vocab.len()];
    //     for (word, word_count) in &self.words {
    //         for &token in word {
    //             result[token as usize] += word_count;
    //         }
    //     }
    //     result
    // }
}

fn substitute_pair(word: &mut Vec<Token>, t0: Token, t1: Token, new_token: Token) {
    let mut i = 0;
    let mut j = 0;
    while i < word.len() {
        if word[i] == t0 && i + 1 < word.len() && word[i + 1] == t1 {
            word[j] = new_token;
            j += 1;
            i += 2;
        } else {
            word[j] = word[i];
            j += 1;
            i += 1;
        }
    }
    word.truncate(j);
}

pub fn word_to_tokens(word: &[u8]) -> Vec<Token> {
    word.iter().map(|&b| b as Token).collect()
}
