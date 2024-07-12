use std::collections::HashMap;
use std::hash::Hash;

use rayon::iter::{FromParallelIterator, IntoParallelIterator, ParallelIterator};

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
    pub fn new_from_word_hashmap(words: HashMap<Word, u64>) -> Self {
        WordCounter {
            words: words
                .into_iter()
                .map(|(word, count)| (word.as_bytes().to_vec(), count))
                .collect(),
        }
    }

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
        let time = std::time::Instant::now();
        let mut words: Vec<_> = self.words.into_iter().collect();
        words.retain(|(_, count)| *count >= MIN_WORD_COUNT);
        let words = words
            .into_iter()
            .map(|(word, count)| (bpe::word_to_tokens(&word), count))
            .collect();
        let res = Bpe::new_and_run(words);
        println!("Elapsed time for into_bpe: {:?}", time.elapsed());
        res
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

impl FromParallelIterator<Word> for WordCounter {
    fn from_par_iter<I: IntoParallelIterator<Item = Word>>(iter: I) -> Self {
        WordCounter::new_from_word_hashmap(count_distinct_parallel(iter.into_par_iter()))
    }
}

fn count_distinct_parallel<K, I>(iter: I) -> HashMap<K, u64>
where
    K: Eq + Hash + Clone + Send + Sync,
    I: ParallelIterator<Item = K>,
{
    iter.fold(
        || HashMap::new(),
        |mut acc, item| {
            *acc.entry(item).or_insert(0) += 1;
            acc
        },
    )
    .reduce(
        || HashMap::new(),
        |mut a, b| {
            for (k, v) in b {
                *a.entry(k).or_insert(0) += v;
            }
            a
        },
    )
}
