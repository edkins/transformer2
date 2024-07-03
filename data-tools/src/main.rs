use split_words::SplitWords;
use word_counter::WordCounter;
use ref_iterator::{AlreadyRefs, RefIterator, Refs};

mod bpe;
mod process_wikitext;
mod process_xml;
mod progress_reader;
mod ref_iterator;
mod split_words;
mod split_words2;
mod tokenize;
mod word_counter;

const MAX_ARTICLES_TO_PROCESS: usize = 5000;

fn main() {
    let filename = "../enwiki-latest-pages-articles-multistream.xml.bz2";
    let file_size = std::fs::metadata(filename).unwrap().len();
    let file = std::fs::File::open(filename).unwrap();
    let progress_file = progress_reader::ProgressReader::new(file, file_size);
    let bfile = std::io::BufReader::new(progress_file);
    let decomp = bzip2::read::MultiBzDecoder::new(bfile);
    let bdecomp = std::io::BufReader::new(decomp);
    
    let word_counter = process_xml::ArticleReader::new(bdecomp)
        .filter_map(process_wikitext::strip_wikitext)
        .take(MAX_ARTICLES_TO_PROCESS)
        .refs()
        .flat_map(|x|SplitWords::new(x).already_refs())
        .collectz::<WordCounter>();

    word_counter.dump();
    let mut bpe = word_counter.into_bpe();
    bpe.run();
    let counts = bpe.get_token_counts();
    println!("Writing dictionary to file");
    bpe.write_to_file("dictionary.txt");
    println!("Dictionary size: {}", counts.len());
    let dict = bpe.into_dictionary();
    for (i,(token, &count)) in dict.iter().zip(counts.iter()).enumerate() {
        word_counter::print_word(i, token, count);
    }
}
