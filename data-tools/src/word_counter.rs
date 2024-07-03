use std::collections::HashMap;

use crate::{
    bpe::{self, Bpe},
    split_words::Word,
};

const MIN_WORD_COUNT: u64 = 2;

#[derive(Default)]
pub struct WordCounter {
    words: HashMap<Vec<u8>, u64>,
}

impl WordCounter {
    pub fn add_word(&mut self, word: Word) {
        self.add_word_bytes(word.as_bytes());
    }

    pub fn add_word_bytes(&mut self, word: &[u8]) {
        let count = self.words.entry(word.to_vec()).or_insert(0);
        *count += 1;
    }

    // pub fn add_document_split_into_words(&mut self, document: &str) {
    //     for word in SplitWords::new(document.as_bytes()) {
    //         self.add_word(word);
    //     }
    // }

    // pub fn dump(&self) {
    //     let mut words: Vec<_> = self.words.iter().collect();
    //     words.sort_by_key(|&(_, count)| std::cmp::Reverse(count));
    //     for (i, (word, &count)) in words.iter().enumerate().take(100) {
    //         print_word(i, word, count);
    //     }
    //     println!("Total number of distinct words: {}", words.len());
    //     println!(
    //         "Total number of words: {}",
    //         words.iter().map(|(_, &count)| count).sum::<u64>()
    //     );
    // }

    pub fn into_bpe(self) -> Bpe {
        let mut words: Vec<_> = self.words.into_iter().collect();
        words.retain(|(_, count)| *count >= MIN_WORD_COUNT);
        let words = words
            .into_iter()
            .map(|(word, count)| (bpe::word_to_tokens(&word), count))
            .collect();
        Bpe::new_and_run(words)
    }
}

// pub fn print_word(i: usize, word: &[u8], count: u64) {
//     if let Ok(string) = std::str::from_utf8(word) {
//         println!("{} {:?}: {}", i, string, count);
//     } else {
//         println!("{} {:?}: {}", i, word, count);
//     }
// }

impl FromIterator<Word> for WordCounter {
    fn from_iter<I: IntoIterator<Item = Word>>(iter: I) -> Self {
        let mut counter = WordCounter::default();
        for word in iter {
            counter.add_word(word)
        }
        counter
    }
}
