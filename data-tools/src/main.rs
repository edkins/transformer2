use std::io::{BufRead, BufReader, Write};
use std::str;

use base64::{prelude::BASE64_STANDARD, Engine};
use byteorder::WriteBytesExt;
use clap::{Parser, Subcommand};
use rand::seq::SliceRandom;
use rand::Rng;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use parallel_xml::{find_all_page_starts, read_page_starts};
use progress_iter::ProgressIter;
use word_counter::WordCounter;

mod bpe;
mod parallel_xml;
mod process_wikitext;
mod process_xml;
mod progress_iter;
mod progress_reader;
mod split_words;
mod tokenize;
mod word_counter;

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
        #[clap(short = 's')]
        split_filename: String,
        #[clap(short = 'o')]
        out_filename: String,
        #[clap(short = 'n', default_value = "1000000")]
        num_articles: usize,
    },
    Tokenize {
        filename: String,
        #[clap(short = 'd')]
        dict_filename: String,
    },
    Print {
        dict_filename: String,
    },
    Split {
        filename: String,
        #[clap(short = 'o')]
        out_filename: String,
    },
}

fn main() {
    let cli = Args::parse();
    let time = std::time::Instant::now();
    let mut rng = rand::thread_rng();

    match cli.subcmd {
        Command::Dictionary {
            filename,
            split_filename,
            out_filename,
            num_articles,
        } => dictionary(
            &mut rng,
            &filename,
            &split_filename,
            &out_filename,
            num_articles,
        ),
        Command::Tokenize {
            filename,
            dict_filename,
        } => tokenize(&filename, &dict_filename),
        Command::Print { dict_filename } => print_dict(&dict_filename),
        Command::Split {
            filename,
            out_filename,
        } => split_xml(&filename, &out_filename),
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
    let progress_file =
        progress_reader::ProgressReader::new(file, std::fs::metadata(filename).unwrap().len());
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

fn print_dict(dict_filename: &str) {
    let dictionary = std::fs::File::open(dict_filename).unwrap();
    for (i, token) in BufReader::new(dictionary)
        .lines()
        .map(|line| BASE64_STANDARD.decode(line.unwrap()).unwrap())
        .enumerate()
    {
        println!("{} {}", i, byte_to_quoted_string(&token));
    }
}

fn dictionary(
    rng: &mut impl Rng,
    filename: &str,
    split_filename: &str,
    out_filename: &str,
    num_articles: usize,
) {
    let articles = read_page_starts(filename, split_filename);
    articles
        .choose_multiple(rng, articles.len()) // choose extra because some will be redirects and hence discarded
        .cloned()
        .collect::<Vec<_>>()
        .par_iter()
        .flat_map_iter(|section| process_xml::ArticleReader::new(section.buf_reader()))
        .filter_map(process_wikitext::strip_wikitext)
        .take_any(num_articles)
        .show_progress(num_articles as u64)
        .flat_map_iter(split_words::split_words)
        .collect::<WordCounter>()
        .into_bpe()
        .write_to_file(out_filename);
}

fn tokenize(filename: &str, dict_filename: &str) {
    let mut tokenizer = tokenize::Tokenizer::from_file(dict_filename);

    let vector = process_xml::ArticleReader::new(decompress(filename))
        .filter_map(process_wikitext::strip_wikitext)
        .flat_map(split_words::split_words)
        .flat_map(|word| tokenizer.tokenize_word_to_bytes(word).to_vec())
        .collect::<Vec<_>>();

    println!("Tokenized {} bytes", vector.len());

    let mut file = std::fs::File::create(TOKENIZED_FILE).unwrap();
    file.write_all(&vector).unwrap();
}
