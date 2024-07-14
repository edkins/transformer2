use std::cell::RefCell;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::str;
use std::sync::Mutex;

use base64::{prelude::BASE64_STANDARD, Engine};
use byteorder::ReadBytesExt;
use clap::{Parser, Subcommand};
use process_xml::Article;
use rayon::iter::{ParallelBridge, ParallelIterator};

use tokenize::TokenizerOutput;
use word_counter::WordCounter;

mod bpe;
mod little_endian;
mod process_wikitext;
mod process_xml;
mod progress_reader;
mod split_words;
mod tokenize;
mod train_test_split;
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
        #[clap(short = 'o')]
        out_filename: String,
    },
    Tokenize {
        filename: String,
        #[clap(short = 'd')]
        dict_filename: String,
        #[clap(short = 'o')]
        out_filename: String,
        #[clap(short = 'p', default_value = "8")]
        parallelism: usize,
    },
    Print {
        dict_filename: String,
    },
    Cat {
        filename: String,
        #[clap(short = 'd')]
        dict_filename: String,
    },
}

fn main() {
    let cli = Args::parse();
    let time = std::time::Instant::now();

    match cli.subcmd {
        Command::Dictionary {
            filename,
            out_filename,
        } => dictionary(&filename, &out_filename),
        Command::Tokenize {
            filename,
            dict_filename,
            out_filename,
            parallelism,
        } => tokenize(&filename, &dict_filename, &out_filename, parallelism),
        Command::Print { dict_filename } => print_dict(&dict_filename),
        Command::Cat {
            filename,
            dict_filename,
        } => {
            cat(&filename, &dict_filename);
        }
    }

    println!("Elapsed time: {:?}", time.elapsed());
}

fn cat(filename: &str, dict_filename: &str) {
    let dictionary = std::fs::File::open(dict_filename).unwrap();
    let dictionary = BufReader::new(dictionary);
    let dictionary = std::iter::once(b"[--SEP--]".to_vec())
        .chain(
            dictionary
                .lines()
                .map(|line| BASE64_STANDARD.decode(line.unwrap()).unwrap()),
        )
        .collect::<Vec<_>>();
    let in_file = std::fs::File::open(filename).unwrap();
    let mut in_file = BufReader::new(in_file);
    let mut out_file = BufWriter::new(std::io::stdout());
    while let Ok(tok) = in_file.read_u16::<byteorder::LittleEndian>() {
        out_file
            .write_all(&dictionary[tok as usize])
            .expect("Failed to write to stdout");
    }
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

fn dictionary(filename: &str, out_filename: &str) {
    let in_file = progress_read_input(filename);
    process_xml::ArticleReader::new(in_file)
        .par_bridge()
        .filter(|a| a.tokenize)
        .map(Article::strip_wikitext)
        .flat_map_iter(|a| split_words::split_words(a.text))
        .collect::<WordCounter>()
        .into_bpe()
        .write_to_file(out_filename);
}

thread_local! {static TOKENIZER: RefCell<tokenize::Tokenizer> = RefCell::new(tokenize::Tokenizer::default())}

fn progress_read_input(
    filename: &str,
) -> BufReader<progress_reader::ProgressReader<std::fs::File>> {
    let length = std::fs::metadata(filename)
        .expect("Failed to get metadata")
        .len();
    let file = std::fs::File::open(filename).expect("Failed to open file");
    let file = progress_reader::ProgressReader::new(file, length);
    BufReader::new(file)
}

fn tokenize(filename: &str, dict_filename: &str, out_filename: &str, parallelism: usize) {
    let in_file = progress_read_input(filename);
    let out_file = Mutex::new(TokenizerOutput::create(out_filename));
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(parallelism)
        .build()
        .expect("Unable to create thread pool");
    pool.install(move || {
        process_xml::ArticleReader::new(in_file)
            .par_bridge()
            .map(Article::strip_wikitext)
            .map(|a| {
                TOKENIZER.with(|tokenizer| {
                    tokenizer
                        .borrow_mut()
                        .initialize_and_tokenize_article(dict_filename, a)
                })
            })
            .for_each(|a| out_file.lock().expect("Poisoned").write_article(a));
        out_file.into_inner().expect("Poisoned").write_metadata();
    });
}
