use std::cell::RefCell;
use std::io::{BufRead, BufReader};
use std::str;
use std::sync::Mutex;

use base64::{prelude::BASE64_STANDARD, Engine};
use clap::{Parser, Subcommand};
use rand::seq::SliceRandom;
use rand::Rng;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use little_endian::LittleEndianStruct;
use parallel_xml::{find_all_page_starts, find_page_starts, read_page_starts};
use progress_iter::ParallelProgressIter;
use word_counter::WordCounter;

mod bpe;
mod little_endian;
mod parallel_xml;
mod process_wikitext;
mod process_xml;
mod progress_iter;
mod progress_reader;
mod split_words;
mod tokenize;
mod word_counter;

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
        #[clap(short = 'o')]
        out_filename: String,
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
            out_filename,
        } => tokenize(
            &filename,
            &dict_filename,
            &out_filename,
        ),
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
    data.write_little_endian(out_file);
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

thread_local! {static TOKENIZER: RefCell<tokenize::Tokenizer> = RefCell::new(tokenize::Tokenizer::default())}

fn tokenize(
    filename: &str,
    dict_filename: &str,
    out_filename: &str,
) {
    let out_file = Mutex::new(std::fs::File::create(out_filename).unwrap());
    find_page_starts(filename)
        .par_iter()
        .flat_map_iter(|section| process_xml::ArticleReader::new(section.buf_reader()))
        .show_progress(24000000) // TODO: how to estimate article count for progress bar?
        .filter_map(process_wikitext::strip_wikitext)
        .map(|text| {
            TOKENIZER.with(|tokenizer| {
                let mut tokenizer = tokenizer.borrow_mut();
                split_words::split_words(text)
                    .flat_map(|word| {
                        tokenizer
                            .initialize_and_tokenize(dict_filename, word)
                            .to_vec()
                    })
                    .chain(std::iter::once(0))
                    .collect::<Vec<_>>()
            })
        })
        .for_each(|vec| {
            vec.write_little_endian(&*out_file.lock().expect("Poisoned."));
        });
}
