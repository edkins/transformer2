use nom::{branch::alt, bytes::complete::{tag, take}, character::complete::alpha1, combinator::{opt, recognize, value}, sequence::preceded, IResult};

pub struct SplitWords<'a> {
    /// The remaining text to split
    text: &'a[u8],
}

impl<'a> SplitWords<'a> {
    pub fn new(text: &'a[u8]) -> Self {
        SplitWords {
            text,
        }
    }
}

fn word(i: &[u8]) -> IResult<&[u8], ()> {
    value((), alpha1)(i)
}

fn maybe_space_word(i: &[u8]) -> IResult<&[u8], ()> {
    // a word optionally preceded by a space
    preceded(opt(tag(" ")), word)(i)
}

fn single_byte(i: &[u8]) -> IResult<&[u8], ()> {
    // match a single byte, not more than one
    // TODO: consume an entire utf-8 character, not just a single byte here
    value((),take(1usize))(i)
}

fn word_like_thing(i: &[u8]) -> IResult<&[u8], &[u8]> {
    // match a word or a single byte
    recognize(alt((maybe_space_word, single_byte)))(i)
}

impl<'a> Iterator for SplitWords<'a> {
    type Item = &'a[u8];

    fn next(&mut self) -> Option<Self::Item> {
        // Return None if we're at the end
        if self.text.is_empty() {
            return None;
        }

        // Otherwise try to consume a word_like_thing
        let (text, wordlike) = word_like_thing(&self.text).expect("Shouldn't really happen");
        self.text = text;

        Some(wordlike)
    }
}

