use std::io::{BufRead, BufReader, Write};
use std::str;

use base64::{prelude::BASE64_STANDARD, Engine};
use clap::{Parser, Subcommand};
use word_counter::WordCounter;

mod bpe;
mod process_wikitext;
mod process_xml;
mod progress_reader;
mod split_words;
mod tokenize;
mod word_counter;

const MAX_ARTICLES_TO_PROCESS: usize = 5000;
const CORPUS_FILENAME: &str = "../enwiki-latest-pages-articles-multistream.xml.bz2";
const DICTIONARY_FILENAME: &str = "dictionary.txt";
const TOKENIZED_FILE: &str = "tokenized.bin";

#[derive(Parser)]
struct Args {
    #[clap(subcommand)]
    subcmd: Command,
}

#[derive(Subcommand, PartialEq, Eq)]
enum Command {
    Dictionary {},
    Tokenize {},
    Print {}
}

fn main() {
    let cli = Args::parse();

    if cli.subcmd == (Command::Print{}) {
        print_dict();
        return;
    }

    let filename = CORPUS_FILENAME;
    let file_size = std::fs::metadata(filename).unwrap().len();
    let file = std::fs::File::open(filename).unwrap();
    let progress_file = progress_reader::ProgressReader::new(file, file_size);
    let bfile = std::io::BufReader::new(progress_file);
    let decomp = bzip2::read::MultiBzDecoder::new(bfile);
    let bdecomp = std::io::BufReader::new(decomp);

    match cli.subcmd {
        Command::Dictionary {} => dictionary(bdecomp),
        Command::Tokenize {} => tokenize(bdecomp),
        _ => {}
    }
}

fn byte_to_quoted_string(bytes: &[u8]) -> String {
    if str::from_utf8(bytes).is_ok() {
        format!("{:?}", str::from_utf8(bytes).unwrap())
    } else {
        format!("{:?}", bytes)
    }
}

fn print_dict() {
    let dictionary = std::fs::File::open(DICTIONARY_FILENAME).unwrap();
    for (i,token) in BufReader::new(dictionary)
        .lines()
        .map(|line| BASE64_STANDARD.decode(line.unwrap()).unwrap())
        .enumerate()
    {
        println!("{} {}", i, byte_to_quoted_string(&token));
    }
}

fn dictionary(bdecomp: impl BufRead) {
    process_xml::ArticleReader::new(bdecomp)
        .filter_map(process_wikitext::strip_wikitext)
        .take(MAX_ARTICLES_TO_PROCESS)
        .flat_map(split_words::split_words)
        .collect::<WordCounter>()
        .into_bpe()
        .write_to_file(DICTIONARY_FILENAME);
}

fn tokenize(bdecomp: impl BufRead) {
    let mut tokenizer = tokenize::Tokenizer::from_file(DICTIONARY_FILENAME);

    let vector = process_xml::ArticleReader::new(bdecomp)
        .filter_map(process_wikitext::strip_wikitext)
        .take(MAX_ARTICLES_TO_PROCESS)
        .flat_map(split_words::split_words)
        .flat_map(|word| tokenizer.tokenize_word_to_bytes(word).to_vec())
        .collect::<Vec<_>>();

    println!("Tokenized {} bytes", vector.len());

    let mut file = std::fs::File::create(TOKENIZED_FILE).unwrap();
    file.write_all(&vector).unwrap();
}
