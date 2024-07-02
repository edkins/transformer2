use process_xml::ArticleError;

mod process_wikitext;
mod process_xml;
mod progress_reader;

const MAX_ARTICLES_TO_PROCESS: usize = 50000;

fn main() {
    let filename = "../enwiki-latest-pages-articles-multistream.xml.bz2";
    let file_size = std::fs::metadata(filename).unwrap().len();
    let file = std::fs::File::open(filename).unwrap();
    let progress_file = progress_reader::ProgressReader::new(file, file_size);
    let bfile = std::io::BufReader::new(progress_file);
    let decomp = bzip2::read::MultiBzDecoder::new(bfile);
    let bdecomp = std::io::BufReader::new(decomp);

    let mut count = 0;
    let time = std::time::Instant::now();
    let articles = process_xml::ArticleReader::new(bdecomp);
    for article in articles {
        match article {
            Ok(article) => {
                count += 1;
                if count % 5000 == 0 {
                    println!("{}: {}", (std::time::Instant::now() - time).as_secs(), count);
                }
                if count == MAX_ARTICLES_TO_PROCESS {
                    break;
                }
                if let Some(plain_article) = process_wikitext::strip_wikitext(&article) {
                    if count < 10 {
                        println!("{plain_article}")
                    }
                }
            }
            Err(ArticleError(e)) => {
                eprintln!("Error reading article {e}");
                return;
            }
        }
    }
    println!("Read {count} articles");
}
