use std::io::BufRead;

use quick_xml::{events::Event, Reader};

pub struct ArticleError(pub String);

pub struct ArticleReader<R> {
    xml_reader: Reader<R>,
}

impl<R: BufRead> ArticleReader<R> {
    pub fn new(reader: R) -> Self {
        ArticleReader {
            xml_reader: Reader::from_reader(reader),
        }
    }
}

impl<R: BufRead> Iterator for ArticleReader<R> {
    type Item = Result<String, ArticleError>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut buf = Vec::new();
        let mut in_text = false;
        let mut article = String::new();
        loop {
            match self.xml_reader.read_event_into(&mut buf) {
                Ok(Event::Start(e)) => {
                    let tag = e.local_name().into_inner();
                    if tag == b"text" {
                        in_text = true;
                    }
                }
                Ok(Event::End(e)) => {
                    let tag = e.local_name().into_inner();
                    if tag == b"text" {
                        return Some(Ok(article));
                    }
                }
                Ok(Event::Text(e)) => {
                    if in_text {
                        article.push_str(&e.unescape().unwrap());
                    }
                }
                Ok(Event::Eof) if !in_text => return None,
                Ok(Event::Eof) => return Some(Err(ArticleError("EOF while in text".to_string()))),
                Ok(_) => {}
                Err(e) => return Some(Err(ArticleError(format!("Error: {}", e)))),
            }
        }
    }
}
