use std::io::BufRead;

use quick_xml::{events::Event, Reader};

use crate::train_test_split::{decide_split, enwiki_url, Split};

pub struct ArticleReader<R> {
    xml_reader: Reader<R>,
}

#[derive(Clone, Debug)]
pub struct Article {
    pub url: String,
    pub text: String,
    pub split: Split,
    pub tokenize: bool,
}

impl<R: BufRead> ArticleReader<R> {
    pub fn new(reader: R) -> Self {
        ArticleReader {
            xml_reader: Reader::from_reader(reader),
        }
    }
}

impl<R: BufRead> Iterator for ArticleReader<R> {
    type Item = Article;

    fn next(&mut self) -> Option<Self::Item> {
        let mut buf = Vec::new();
        let mut in_text = false;
        let mut in_title = false;
        let mut text = String::new();
        let mut title = String::new();
        let mut redirect = false;
        loop {
            match self.xml_reader.read_event_into(&mut buf) {
                Ok(Event::Start(e)) => match e.local_name().into_inner() {
                    b"text" => {
                        in_text = true;
                    }
                    b"title" => {
                        in_title = true;
                    }
                    // b"redirect" => {
                    //     redirect = true;
                    // }
                    _ => {}
                },
                Ok(Event::Empty(e)) => if e.local_name().into_inner() == b"redirect" {
                    redirect = true;
                }
                Ok(Event::End(e)) => {
                    match e.local_name().into_inner() {
                        b"text" => {
                            if redirect || title.starts_with("Wikipedia:") {
                                // don't return articles if they're redirects or Wikipedia project pages
                                in_text = false;
                                in_title = false;
                                text = String::new();
                                title = String::new();
                                redirect = false;
                            } else {
                                let url = enwiki_url(&title);
                                let (split, tokenize) = decide_split(&url);
                                return Some(Article {
                                    url,
                                    text,
                                    split,
                                    tokenize,
                                });
                            }
                        }
                        b"title" => {
                            in_title = false;
                        }
                        _ => {}
                    }
                }
                Ok(Event::Text(e)) => {
                    if in_text {
                        text.push_str(&e.unescape().unwrap());
                    }
                    if in_title {
                        title.push_str(&e.unescape().unwrap());
                    }
                }
                Ok(Event::Eof) if !in_text => return None,
                Ok(Event::Eof) => {
                    println!("EOF while in text");
                    return None;
                }
                Ok(_) => {}
                Err(e) => {
                    println!("{}", e);
                    return None;
                }
            }
        }
    }
}
