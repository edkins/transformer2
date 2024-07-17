use std::cell::RefCell;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::str;
use std::sync::Mutex;

use base64::{prelude::BASE64_STANDARD, Engine};
use byteorder::ReadBytesExt;
use clap::{Parser, Subcommand};
use process_xml::Article;
use rayon::iter::{ParallelBridge, ParallelIterator};

use serde::Deserialize;
use tokenize::{write_condensed, ArticleMetadata, TokenizedArticle, TokenizerOutput};
use train_test_split::Split;
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
        #[clap(short = 'n')]
        n_tokens: usize,
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
    Condense {
        filename: String,
    },
    Print {
        dict_filename: String,
    },
    Cat {
        filename: String,
        #[clap(short = 'd')]
        dict_filename: String,
    },
    Clean {
        filename: String,
    },
    Retokenize {
        filename: String,
        #[clap(short = 'o')]
        out_filename: String,
    },
}

fn main() {
    let cli = Args::parse();
    let time = std::time::Instant::now();

    match cli.subcmd {
        Command::Dictionary {
            filename,
            out_filename,
            n_tokens,
        } => dictionary(&filename, &out_filename, n_tokens),
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
        } => cat(&filename, &dict_filename),
        Command::Condense { filename } => condense(&filename),
        Command::Clean { filename } => clean(&filename),
        Command::Retokenize { filename, out_filename } => retokenize(&filename, &out_filename),
    }

    println!("Elapsed time: {:?}", time.elapsed());
}

#[derive(Deserialize)]
struct AllMetadataIn {
    train: Vec<ArticleMetadata>,
    validation: Vec<ArticleMetadata>,
    test: Vec<ArticleMetadata>,
}

fn retokenize(filename: &str, out_filename: &str) {
    let dictionary = BufReader::new(File::open(format!("{}.dictionary", filename)).unwrap());
    let dictionary = std::iter::once(b"[--SEP--]".to_vec())
        .chain(
            dictionary
            .lines()
            .map(|line| BASE64_STANDARD.decode(line.unwrap()).unwrap()),
        )
        .collect::<Vec<_>>();

    let mut tokenizer = tokenize::Tokenizer::default();
    tokenizer.initialize(&format!("{}.dictionary", out_filename));
    
    let mut output = TokenizerOutput::create(out_filename);
    for split_enum in [Split::Train, Split::Validation, Split::Test] {
        let split = split_enum.to_str();
        println!("Processing split {}", split);
        let mut in_file = progress_read_input(&format!("{}.{}", filename, split));
        let mut article_in = Vec::new();
        while let Ok(tok) = in_file.read_u16::<byteorder::LittleEndian>() {
            if tok != 0 {
                article_in.extend_from_slice(&dictionary[tok as usize]);
                continue;
            }
            let mut article_out = Vec::new();
            let mut new_article = Vec::new();
            std::mem::swap(&mut article_in, &mut new_article);
            for word in split_words::split_words(String::from_utf8(new_article).expect("utf-8 error")) {
                article_out.extend_from_slice(tokenizer.tokenize_word(word));
            }
            article_out.push(0);
            output.write_article(TokenizedArticle {
                url: String::new(),  // TODO: this shouldn't really be here. It gets ignored by the later stages
                split: split_enum,
                tokens: article_out,
            });
        }
    };
    output.write_metadata();
}

fn clean(filename: &str) {
    let in_file = BufReader::new(File::open(filename).unwrap());
    process_xml::ArticleReader::new(in_file)
        .skip(10_000)
        .take(10_000)
        .for_each(|a| {
            //a.text.chars().take(3000).for_each(|c| print!("{}", c));
            //println!("\n-----------");
            a.strip_wikitext()
                .text
                .chars()
                .take(2000)
                .for_each(|c| print!("{}", c));
            println!("\n-----------");
        });
}

fn condense(filename: &str) {
    let metadata = File::open(format!("{}.metadata", filename)).unwrap();
    let metadata: AllMetadataIn = serde_json::from_reader(BufReader::new(metadata)).unwrap();
    write_condensed(&metadata.train, &format!("{}.metadata.train", filename));
    write_condensed(
        &metadata.validation,
        &format!("{}.metadata.validation", filename),
    );
    write_condensed(&metadata.test, &format!("{}.metadata.test", filename));
}

fn cat(filename: &str, dict_filename: &str) {
    let dictionary = File::open(dict_filename).unwrap();
    let dictionary = BufReader::new(dictionary);
    let dictionary = std::iter::once(b"[--SEP--]".to_vec())
        .chain(
            dictionary
                .lines()
                .map(|line| BASE64_STANDARD.decode(line.unwrap()).unwrap()),
        )
        .collect::<Vec<_>>();
    let in_file = File::open(filename).unwrap();
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
    let dictionary = File::open(dict_filename).unwrap();
    for (i, token) in BufReader::new(dictionary)
        .lines()
        .map(|line| BASE64_STANDARD.decode(line.unwrap()).unwrap())
        .enumerate()
    {
        println!("{} {}", i + 1, byte_to_quoted_string(&token));
    }
}

fn dictionary(filename: &str, out_filename: &str, n_tokens: usize) {
    let in_file = progress_read_input(filename);
    process_xml::ArticleReader::new(in_file)
        .par_bridge()
        .filter(|a| a.tokenize)
        .map(Article::strip_wikitext)
        .flat_map_iter(|a| split_words::split_words(a.text))
        .collect::<WordCounter>()
        .into_bpe(n_tokens)
        .write_to_file(out_filename);
}

thread_local! {static TOKENIZER: RefCell<tokenize::Tokenizer> = RefCell::new(tokenize::Tokenizer::default())}

fn progress_read_input(filename: &str) -> BufReader<progress_reader::ProgressReader<File>> {
    let length = std::fs::metadata(filename)
        .expect("Failed to get metadata")
        .len();
    let file = File::open(filename).expect("Failed to open file");
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
