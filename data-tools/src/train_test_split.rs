use std::hash::Hasher;

use rustc_hash::FxHasher;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Split {
    Train,
    Validation,
    Test,
}

impl Split {
    pub fn to_n(self) -> usize {
        match self {
            Split::Train => 0,
            Split::Validation => 1,
            Split::Test => 2,
        }
    }

    pub fn from_n(n: usize) -> Self {
        match n {
            0 => Split::Train,
            1 => Split::Validation,
            2 => Split::Test,
            _ => panic!("Invalid split number: {}", n),
        }
    }

    pub fn to_str(self) -> &'static str {
        match self {
            Split::Train => "train",
            Split::Validation => "validation",
            Split::Test => "test",
        }
    }
}

/// Decide which split a URL should be in.
/// Also decide whether it should be used for tokenization
pub fn decide_split(url: &str) -> (Split, bool) {
    let mut hasher = FxHasher::default();
    hasher.write(url.as_bytes());
    match hasher.finish() % 20 {
        0 => (Split::Validation, false),
        1 => (Split::Test, false),
        2 => (Split::Train, true),
        _ => (Split::Train, false),
    }
}

pub fn enwiki_url(title: &str) -> String {
    let title = title.replace(' ', "_");
    let title = url_escape::encode_component(&title);
    format!("https://en.wikipedia.org/wiki/{}", title)
}
