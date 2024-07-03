use std::collections::HashMap;

pub type Token = u16;

pub struct Tokenizer {
    cache: HashMap<Vec<u8>, Vec<Token>>,
}

impl Tokenizer {
    pub fn new(dictionary: Vec<Vec<u8>>) -> Self {
        let mut cache = HashMap::new();
        for (i, word) in dictionary.into_iter().enumerate() {
            cache.insert(word.clone(), vec![i as Token]);
        }
        Tokenizer {
            cache,
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

    fn split_into_token_and_remainder<'a>(&'a self, text: &'a[u8]) -> (Token, &'a[u8]) {
        let mut i = text.len();
        loop {
            if i == 0 {
                panic!("No token found for {:?}", text);
            }
            if let Some(token) = self.single_token(text) {
                return (token, &text[i..]);
            }
            i -= 1;
        }
    }

    fn split_into_tokens<'a>(&'a self, text: &'a[u8]) -> Vec<Token> {
        let mut tokens = vec![];
        let mut text = text;
        while !text.is_empty() {
            let (token, remainder) = self.split_into_token_and_remainder(text);
            tokens.push(token);
            text = remainder;
        }
        tokens
    }

    pub fn tokenize_word<'a>(&'a mut self, word: &[u8]) -> &'a[Token] {
        if !self.cache.contains_key(word) {
            let tokens = self.split_into_tokens(word);
            self.cache.insert(word.to_vec(), tokens);
        }
        self.cache.get(word).unwrap()
    }
}