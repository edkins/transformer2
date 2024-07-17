use byteorder::ReadBytesExt;
use std::io::BufRead;

pub struct SplitTokenizedArticles<F> {
    in_file: F,
    dictionary: Vec<Vec<u8>>,
}

impl<F: BufRead> SplitTokenizedArticles<F> {
    pub fn new(in_file: F, dictionary: Vec<Vec<u8>>) -> Self {
        SplitTokenizedArticles {
            in_file,
            dictionary,
        }
    }
}

impl<F: BufRead> Iterator for SplitTokenizedArticles<F> {
    type Item = String;

    fn next(&mut self) -> Option<Self::Item> {
        let mut result = Vec::new();
        loop {
            match self.in_file.read_u16::<byteorder::LittleEndian>() {
                Ok(0) => return Some(String::from_utf8(result).expect("Invalid UTF-8")),
                Ok(token @1..=65535) => result.extend_from_slice(&self.dictionary[token as usize]),
                Err(_) => return None,
            }
        }
    }
}
