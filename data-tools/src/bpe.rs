use std::{
    collections::{BTreeSet, HashMap},
    io::Write,
};

use base64::{prelude::BASE64_STANDARD, Engine};

type Token = u32;

struct PairCounter {
    map: HashMap<(Token, Token), (i64, Vec<usize>)>,
    set: BTreeSet<(i64, (Token, Token))>,
}

impl PairCounter {
    fn new_from_words(words: &[(Vec<Token>, u64)]) -> Self {
        let mut result = PairCounter {
            map: HashMap::new(),
            set: BTreeSet::new(),
        };
        for (i, (word, word_count)) in words.iter().enumerate() {
            result.add_token_pairs(word, *word_count, i);
        }
        result
    }

    fn find_most_common_pair(&self) -> Option<(Token, Token)> {
        self.set.iter().next_back().map(|(_, k)| *k)
    }

    pub fn adjust(&mut self, t0: Token, t1: Token, word_id: usize, delta: i64) {
        let key = (t0, t1);
        let new_value = if let Some(value) = self.map.get(&key) {
            self.set.remove(&(value.0, (t0, t1)));
            value.0 + delta
        } else {
            delta
        };

        match new_value {
            ..=-1 => panic!("Negative pair count"),
            0 => {
                self.map.remove(&key);
            }
            1.. => {
                let entry = self.map.entry(key).or_insert_with(|| (0, Vec::new()));
                entry.0 += delta;
                if entry.1.last() != Some(&word_id) {
                    entry.1.push(word_id);
                }
                self.set.insert((new_value, key));
            }
        }
    }

    fn add_token_pairs(&mut self, word: &[Token], word_count: u64, word_id: usize) {
        let mut prev = word[0];
        for &token in &word[1..] {
            self.adjust(prev, token, word_id, word_count as i64);
            prev = token;
        }
    }

    fn subtract_token_pairs(&mut self, word: &[Token], word_count: u64, _word_id: usize) {
        let mut prev = word[0];
        for &token in &word[1..] {
            self.adjust(prev, token, _word_id, -(word_count as i64));
            prev = token;
        }
    }

    fn get_words_containing_token_pair(&self, t0: Token, t1: Token) -> Vec<usize> {
        if let Some(v) = self.map.get(&(t0, t1)) {
            v.1.clone()
        } else {
            vec![]
        }
    }
}

pub struct Bpe {
    words: Vec<(Vec<Token>, u64)>,
    token_vocab: Vec<Vec<u8>>,
    pairs: PairCounter,
    n_tokens: usize,
}

impl Bpe {
    pub fn new_and_run(words: Vec<(Vec<Token>, u64)>, n_tokens: usize) -> Self {
        let mut result = Bpe {
            token_vocab: (0..=255).map(|x| [x].to_vec()).collect(),
            pairs: PairCounter::new_from_words(&words),
            words,
            n_tokens,
        };

        result.run();
        result
    }

    fn substitute_pair(&mut self, t0: Token, t1: Token, new_token: Token) {
        // let mut updated_count = 0;

        let word_ids = self.pairs.get_words_containing_token_pair(t0, t1);
        //println!("Token: {}. Found {} words", new_token, words.len());
        for i in word_ids {
            if contains_pair(&self.words[i].0, t0, t1) {
                //println!("New token: {}. Relevant word = {}", new_token, i);
                let (word, word_count) = &mut self.words[i];
                self.pairs.subtract_token_pairs(word, *word_count, i);
                substitute_pair_in_word(word, t0, t1, new_token);
                self.pairs.add_token_pairs(word, *word_count, i);
                // updated_count += 1;
            }
        }
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
        while self.token_vocab.len() < self.n_tokens - 1 {
            self.step();
            if self.token_vocab.len() % 100 == 0 {
                println!("Dictionary size: {}", self.token_vocab.len());
            }
        }
    }

    pub fn write_to_file(&self, filename: &str) {
        let mut file = std::fs::File::create(filename).unwrap();
        for token in &self.token_vocab {
            file.write_all(BASE64_STANDARD.encode(token).as_bytes())
                .unwrap();
            file.write_all(b"\n").unwrap();
        }
    }
}

fn contains_pair(word: &[Token], t0: Token, t1: Token) -> bool {
    for i in 0..word.len() - 1 {
        if word[i] == t0 && word[i + 1] == t1 {
            return true;
        }
    }
    false
}

fn substitute_pair_in_word(word: &mut Vec<Token>, t0: Token, t1: Token, new_token: Token) {
    let mut i = 0; // where we're reading from
    let mut j = 0; // where we're writing to
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
