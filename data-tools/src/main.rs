use word_counter::WordCounter;

mod bpe;
mod process_wikitext;
mod process_xml;
mod progress_reader;
mod split_words;
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

    process_xml::ArticleReader::new(bdecomp)
        .filter_map(process_wikitext::strip_wikitext)
        .take(MAX_ARTICLES_TO_PROCESS)
        .flat_map(split_words::split_words)
        .collect::<WordCounter>()
        .into_bpe()
        .write_to_file("dictionary.txt");
}
