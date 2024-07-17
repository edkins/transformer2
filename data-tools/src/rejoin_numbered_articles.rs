use std::{cmp::Reverse, collections::BinaryHeap};

use crate::{tokenize::{TokenizedArticle, TokenizerOutput}, train_test_split::Split};

pub struct Rejoiner {
    sink: TokenizerOutput,
    next: [usize;3],
    queue: [BinaryHeap<Reverse<(usize, Vec<u16>)>>;3],
}

impl Rejoiner {
    pub fn new (sink: TokenizerOutput) -> Self {
        Rejoiner {
            sink,
            next: [0;3],
            queue: [BinaryHeap::new(), BinaryHeap::new(), BinaryHeap::new()],
        }
    }

    pub fn add(&mut self, number: usize, tokens: Vec<u16>, split: Split) {
        let i = split.to_n();
        self.queue[i].push(Reverse((number, tokens)));
        while !self.queue[i].is_empty() && self.queue[i].peek().unwrap().0.0 == self.next[i] {
            let next = self.queue[i].pop().unwrap().0;
            self.sink.write_article(TokenizedArticle {
                url: String::new(),  // TODO: remove this field
                split,
                tokens: next.1,
            });
            self.next[i] += 1;
        }
    }

    pub fn finish(self) {
        for i in 0..3 {
            if !self.queue[i].is_empty() {
                panic!("Missing articles in split {}", i);
            }
        }
        self.sink.write_metadata();
    }
}
