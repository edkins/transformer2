use nom::{
    branch::alt,
    bytes::complete::tag,
    character::complete::{alpha1, anychar},
    combinator::{opt, recognize, value},
    sequence::preceded,
    IResult,
};

use crate::ref_iterator::RefIterator;

pub struct SplitWords {
    /// The remaining text to split
    text: String,
    pos: usize,
}

impl SplitWords {
    pub fn new(text: String) -> Self {
        SplitWords { text, pos:0 }
    }
}

fn word(i: &str) -> IResult<&str, ()> {
    value((), alpha1)(i)
}

fn maybe_space_word(i: &str) -> IResult<&str, ()> {
    // a word optionally preceded by a space
    preceded(opt(tag(" ")), word)(i)
}

fn single_char(i: &str) -> IResult<&str, ()> {
    // match a single character, which may be more than one utf-8 byte
    value((), anychar)(i)
}

fn word_like_thing(i: &str) -> IResult<&str, &str> {
    // match a word or a single byte
    recognize(alt((maybe_space_word, single_char)))(i)
}

impl RefIterator for SplitWords {
    type Item = str;

    fn next_and<T,G>(&mut self, g: G) -> Option<T>
    where
        G: FnOnce(&Self::Item) -> T
    {
        // Return None if we're at the end
        if self.pos == self.text.len() {
            return None;
        }

        // Otherwise try to consume a word_like_thing
        let (_, wordlike) = word_like_thing(&self.text).expect("Shouldn't really happen");
        self.pos += wordlike.len();

        Some(g(wordlike))
    }
}
