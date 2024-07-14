use std::{
    collections::HashMap,
    fs::File,
    io::{BufRead, BufReader, BufWriter},
};

use base64::{prelude::BASE64_STANDARD, Engine};
use serde::Serialize;

use crate::little_endian::LittleEndianStruct;
use crate::{
    process_xml::Article,
    split_words::{SplitWords, Word},
    train_test_split::Split,
};

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

pub struct TokenizedArticle {
    pub url: String,
    pub split: Split,
    pub tokens: Vec<u16>,
}

impl Tokenizer {
    pub fn initialize_and_tokenize_article(
        &mut self,
        filename: &str,
        article: Article,
    ) -> TokenizedArticle {
        if self.cache.is_empty() {
            self.initialize(filename);
        }
        let mut tokens = Vec::new();
        for word in SplitWords::new(article.text) {
            tokens.extend(self.tokenize_word(word).iter().cloned());
        }
        tokens.push(0);
        TokenizedArticle {
            url: article.url,
            split: article.split,
            tokens,
        }
    }
}

#[derive(Serialize)]
struct ArticleMetadata {
    pub url: String,
    pub token_start: u64,
    pub token_end: u64,
}

#[derive(Serialize)]
struct AllMetadata<'a> {
    train: &'a [ArticleMetadata],
    validation: &'a [ArticleMetadata],
    test: &'a [ArticleMetadata],
}

pub struct TokenizerOutput {
    files: [BufWriter<File>; 3],
    lengths: [u64; 3],
    metadata: [Vec<ArticleMetadata>; 3],
    metadata_filename: String,
}

impl TokenizerOutput {
    pub fn create(filename: &str) -> Self {
        TokenizerOutput {
            files: [open(filename, 0), open(filename, 1), open(filename, 2)],
            lengths: [0, 0, 0],
            metadata: [vec![], vec![], vec![]],
            metadata_filename: format!("{}.metadata", filename),
        }
    }

    pub fn write_article(&mut self, article: TokenizedArticle) {
        let split = article.split.to_n();
        let length = article.tokens.len() as u64 * 2;
        let metadata = ArticleMetadata {
            url: article.url,
            token_start: self.lengths[split],
            token_end: self.lengths[split] + length,
        };
        article.tokens.write_little_endian(&mut self.files[split]);
        self.metadata[split].push(metadata);
        self.lengths[split] += length;
    }

    pub fn write_metadata(self) {
        let metadata_file =
            File::create(self.metadata_filename).expect("Failed to create metadata file");
        serde_json::to_writer(
            metadata_file,
            &AllMetadata {
                train: &self.metadata[Split::Train.to_n()],
                validation: &self.metadata[Split::Validation.to_n()],
                test: &self.metadata[Split::Test.to_n()],
            },
        )
        .expect("Failed to serialize metadata");
    }
}

fn open(filename: &str, split_n: usize) -> BufWriter<File> {
    let filename = format!("{}.{}", filename, Split::from_n(split_n).to_str());
    let out_file = File::create(filename).expect("Failed to create output file");
    BufWriter::new(out_file)
}
