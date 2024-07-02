use std::collections::HashMap;

use crate::split_words::SplitWords;

pub struct WordCounter {
    words: HashMap<Vec<u8>, u64>,
}

impl WordCounter {
    pub fn new() -> Self {
        WordCounter {
            words: HashMap::new(),
        }
    }

    pub fn add_word(&mut self, word: &[u8]) {
        let count = self.words.entry(word.to_vec()).or_insert(0);
        *count += 1;
    }

    pub fn add_document_split_into_words(&mut self, document: &str) {
        for word in SplitWords::new(document.as_bytes()) {
            self.add_word(word);
        }
    }

    pub fn dump(&self) {
        let mut words: Vec<_> = self.words.iter().collect();
        words.sort_by_key(|&(_, count)| std::cmp::Reverse(count));
        for (i,(word, &count)) in words.iter().enumerate().take(100) {
            print_word(i, word, count);
        }
        // Now show every thousandth word without enumerating every single item
        // let mut i = 0;
        // while i < words.len() {
        //     if i >= 100 {
        //         print_word(i, &words[i].0, *words[i].1);
        //     }
        //     i += 1000;
        // }
        println!("Total number of words: {}", words.len());
    }
}

fn print_word(i: usize, word: &[u8], count: u64) {
    if let Ok(string) = std::str::from_utf8(word) {
        println!("{} {:?}: {}", i, string, count);
    } else {
        println!("{} {:?}: {}", i, word, count);
    }
}
