use nom::{
    branch::alt,
    bytes::complete::{tag, take_while_m_n},
    character::complete::anychar,
    combinator::{opt, recognize, value},
    sequence::preceded,
    IResult,
};

const MAX_WORD_LENGTH: usize = 24;

pub struct Word {
    length: usize,
    bytes: [u8;MAX_WORD_LENGTH],
}

impl Word {
    pub fn from_str(s: &str) -> Self {
        let mut bytes = [0;MAX_WORD_LENGTH];
        let length = s.len();
        if length > MAX_WORD_LENGTH {
            panic!("Word too long: {}", s);
        }
        bytes[0..length].copy_from_slice(s.as_bytes());
        Word { length, bytes }
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes[0..self.length]
    }
}

pub struct SplitWords {
    /// The text to split
    text: String,
    pos: usize,
}

impl SplitWords {
    pub fn new(text: String) -> Self {
        SplitWords { text, pos: 0 }
    }
}

fn word(i: &str) -> IResult<&str, ()> {
    value((), take_while_m_n(1, MAX_WORD_LENGTH, |c:char| c.is_ascii_alphabetic()))(i)
}

fn space_word(i: &str) -> IResult<&str, ()> {
    // a word preceded by a space
    preceded(tag(" "), value((), take_while_m_n(1, MAX_WORD_LENGTH - 1, |c:char| c.is_ascii_alphabetic())))(i)
}

fn single_char(i: &str) -> IResult<&str, ()> {
    // match a single character, which may be more than one utf-8 byte
    value((), anychar)(i)
}

fn word_like_thing(i: &str) -> IResult<&str, &str> {
    // match a word or a single byte
    recognize(alt((word, space_word, single_char)))(i)
}

impl Iterator for SplitWords {
    type Item = Word;

    fn next(&mut self) -> Option<Self::Item> {
        // Return None if we're at the end
        if self.pos >= self.text.len(){
            return None;
        }

        // Otherwise try to consume a word_like_thing
        let (_, wordlike) = word_like_thing(&self.text[self.pos..]).expect("Shouldn't really happen");
        self.pos += wordlike.len();

        Some(Word::from_str(wordlike))
    }
}

pub fn split_words(text: String) -> impl Iterator<Item = Word> {
    SplitWords::new(text)
}