use nom::{
    branch::alt,
    bytes::complete::tag,
    character::complete::{alpha1, anychar},
    combinator::{opt, recognize, value},
    sequence::preceded,
    IResult,
};

pub struct SplitWords<'a> {
    /// The remaining text to split
    text: &'a str,
}

impl<'a> SplitWords<'a> {
    pub fn new(text: &'a str) -> Self {
        SplitWords { text }
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

impl<'a> Iterator for SplitWords<'a> {
    type Item = &'a str;

    fn next(&mut self) -> Option<&'a str> {
        // Return None if we're at the end
        if self.text.is_empty() {
            return None;
        }

        // Otherwise try to consume a word_like_thing
        let (rest, wordlike) = word_like_thing(self.text).expect("Shouldn't really happen");
        self.text = rest;

        Some(wordlike)
    }
}
