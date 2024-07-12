use std::io::{BufRead, BufReader, Write};
use std::str;

use base64::{prelude::BASE64_STANDARD, Engine};
use byteorder::WriteBytesExt;
use clap::{Parser, Subcommand};
use parallel_xml::{find_all_page_starts, find_page_starts};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use word_counter::WordCounter;

mod bpe;
mod parallel_xml;
mod process_wikitext;
mod process_xml;
mod progress_reader;
mod split_words;
mod tokenize;
mod word_counter;

const MAX_ARTICLES_TO_PROCESS: usize = 1000000;
const DICTIONARY_FILENAME: &str = "dictionary.txt";
const TOKENIZED_FILE: &str = "tokenized.bin";

#[derive(Parser)]
struct Args {
    #[clap(subcommand)]
    subcmd: Command,
}

#[derive(Subcommand, PartialEq, Eq)]
enum Command {
    Dictionary {
        filename: String,
    },
    Tokenize {
        filename: String,
    },
    Print {},
    Split {
        filename: String,
        #[clap(short='o')]
        out_filename: String,
    }
}

fn main() {
    let cli = Args::parse();
    let time = std::time::Instant::now();

    match cli.subcmd {
        Command::Dictionary { filename } => dictionary(&filename),
        Command::Tokenize { filename } => tokenize(&filename),
        Command::Print{} => print_dict(),
        Command::Split{ filename, out_filename } => split_xml(&filename, &out_filename),
    }

    println!("Elapsed time: {:?}", time.elapsed());
}

fn split_xml(filename: &str, out_filename: &str) {
    let data = find_all_page_starts(filename);
    let out_file = std::fs::File::create(out_filename).unwrap();
    let mut out_buf = std::io::BufWriter::new(out_file);
    for num in &data {
        out_buf.write_u64::<byteorder::LittleEndian>(*num).unwrap();
    }
}

fn decompress(filename: &str) -> impl BufRead {
    let file = std::fs::File::open(filename).unwrap();
    let progress_file = progress_reader::ProgressReader::new(file, std::fs::metadata(filename).unwrap().len());
    let bfile = std::io::BufReader::new(progress_file);
    let decomp = bzip2::read::MultiBzDecoder::new(bfile);
    std::io::BufReader::new(decomp)
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

fn dictionary(filename: &str) {
    find_page_starts(filename)
        .par_iter()
        .flat_map_iter(|section| process_xml::ArticleReader::new(section.buf_reader()))
        .filter_map(process_wikitext::strip_wikitext)
        .take_any(MAX_ARTICLES_TO_PROCESS)
        .flat_map_iter(split_words::split_words)
        .collect::<WordCounter>()
        .into_bpe()
        .write_to_file(DICTIONARY_FILENAME);
}

fn tokenize(filename: &str) {
    let mut tokenizer = tokenize::Tokenizer::from_file(DICTIONARY_FILENAME);

    let vector = process_xml::ArticleReader::new(decompress(filename))
        .filter_map(process_wikitext::strip_wikitext)
        .take(MAX_ARTICLES_TO_PROCESS)
        .flat_map(split_words::split_words)
        .flat_map(|word| tokenizer.tokenize_word_to_bytes(word).to_vec())
        .collect::<Vec<_>>();

    println!("Tokenized {} bytes", vector.len());

    let mut file = std::fs::File::create(TOKENIZED_FILE).unwrap();
    file.write_all(&vector).unwrap();
}
