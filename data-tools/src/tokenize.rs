use std::{
    collections::HashMap,
    io::{BufRead, BufReader},
};

use base64::{prelude::BASE64_STANDARD, Engine};

use crate::split_words::Word;

pub type Token = u16;

#[derive(Default)]
pub struct Tokenizer {
    cache: HashMap<Vec<u8>, Vec<Token>>,
}

impl Tokenizer {
    pub fn initialize(&mut self, filename: &str) {
        let dictionary = std::fs::File::open(filename).unwrap();
        for (i, word) in BufReader::new(dictionary)
            .lines()
            .map(|line| BASE64_STANDARD.decode(line.unwrap()).unwrap())
            .enumerate()
        {
            self.cache.insert(word.clone(), vec![(i + 1) as Token]);
        }
    }

    fn single_token(&self, text: &[u8]) -> Option<Token> {
        if let Some(tokens) = self.cache.get(text) {
            if tokens.len() == 1 {
                return Some(tokens[0]);
            }
        }
        None
    }

    fn split_into_token_and_remainder<'a>(&'a self, text: &'a [u8]) -> (Token, &'a [u8]) {
        let mut i = text.len();
        loop {
            if i == 0 {
                panic!("No token found for {:?}", text);
            }
            if let Some(token) = self.single_token(&text[..i]) {
                return (token, &text[i..]);
            }
            i -= 1;
        }
    }

    fn split_into_tokens<'a>(&'a self, text: &'a [u8]) -> Vec<Token> {
        let mut tokens = vec![];
        let mut text = text;
        while !text.is_empty() {
            let (token, remainder) = self.split_into_token_and_remainder(text);
            tokens.push(token);
            text = remainder;
        }
        tokens
    }

    pub fn initialize_and_tokenize(&mut self, filename: &str, word: Word) -> &[Token] {
        if self.cache.is_empty() {
            self.initialize(filename);
        }
        self.tokenize_word(word)
    }

    pub fn tokenize_word(&mut self, word: Word) -> &[Token] {
        self.tokenize_word_from_bytes(word.as_bytes())
    }

    pub fn tokenize_word_from_bytes<'a>(&'a mut self, word: &[u8]) -> &'a [Token] {
        if !self.cache.contains_key(word) {
            let tokens = self.split_into_tokens(word);
            self.cache.insert(word.to_vec(), tokens);
        }
        self.cache.get(word).unwrap()
    }
}
