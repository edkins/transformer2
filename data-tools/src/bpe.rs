use std::{collections::HashMap, io::Write};

use base64::{prelude::BASE64_STANDARD, Engine};

type Token = u32;

const NUM_TOKENS_TO_GENERATE: usize = 2000;

struct PairCounter(HashMap<(Token, Token), u64>);

impl PairCounter {
    fn new_from_words(words: &[(Vec<Token>, u64)]) -> Self {
        let mut result = PairCounter(HashMap::new());
        for (word, word_count) in words {
            result.add_token_pairs(word, *word_count);
        }
        result
    }

    fn find_most_common_pair(&self) -> Option<(Token, Token)> {
        self.0.iter().max_by_key(|&(_, count)| count).map(|(&pair, _)| pair)
    }

    fn add_token_pairs(&mut self, word: &[Token], word_count: u64) {
        let mut prev = word[0];
        for &token in &word[1..] {
            let pair = (prev, token);
            let count = self.0.entry(pair).or_insert(0);
            *count += word_count;
            prev = token;
        }
    }

    fn sub_token_pairs(&mut self, word: &[Token], word_count: u64) {
        let mut prev = word[0];
        for &token in &word[1..] {
            let pair = (prev, token);
            let count = self.0.get_mut(&pair).unwrap();
            *count -= word_count;
            if *count == 0 {
                self.0.remove(&pair);
            }
            prev = token;
        }
    }
}

pub struct Bpe {
    words: Vec<(Vec<Token>, u64)>,
    token_vocab: Vec<Vec<u8>>,
    pairs: PairCounter,
}

impl Bpe {
    pub fn new_and_run(words: Vec<(Vec<Token>, u64)>) -> Self {
        let mut result = Bpe {
            token_vocab: (0..=255).map(|x| [x].to_vec()).collect(),
            pairs: PairCounter::new_from_words(&words),
            words,
        };

        result.run();
        result
    }

    fn substitute_pair(&mut self, t0: Token, t1: Token, new_token: Token) {
        let mut updated_count = 0;
        for (word, word_count) in &mut self.words {
            if contains_pair(word, t0, t1) {
                self.pairs.sub_token_pairs(word, *word_count);
                substitute_pair(word, t0, t1, new_token);
                self.pairs.add_token_pairs(word, *word_count);
                updated_count += 1;
            }
        }
        println!("Token: {}. Updated {}/{} words", new_token, updated_count, self.words.len());
    }

    fn step(&mut self) -> bool {
        if let Some((t0, t1)) = self.pairs.find_most_common_pair() {
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
            file.write_all(BASE64_STANDARD.encode(token).as_bytes())
                .unwrap();
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

fn contains_pair(word: &[Token], t0: Token, t1: Token) -> bool {
    let mut i = 0;
    while i < word.len() {
        if word[i] == t0 && i + 1 < word.len() && word[i + 1] == t1 {
            return true;
        }
        i += 1;
    }
    false
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
