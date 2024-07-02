use std::io::BufRead;

use process_xml::ArticleError;

mod process_xml;
mod progress_reader;

fn main() {
    let filename = "../enwiki-latest-pages-articles-multistream.xml.bz2";
    let file_size = std::fs::metadata(filename).unwrap().len();
    let file = std::fs::File::open(filename).unwrap();
    let progress_file = progress_reader::ProgressReader::new(file, file_size);
    let decomp = bzip2::read::MultiBzDecoder::new(progress_file);
    let bdecomp = std::io::BufReader::new(decomp);

    let time = std::time::Instant::now();

    for (i,line) in bdecomp.lines().enumerate() {
        if i % 1000000 == 0 {
            println!("{}: {}", (std::time::Instant::now() - time).as_secs(), i);
        }
    }
    return;

    let articles = process_xml::ArticleReader::new(bdecomp);

    let mut count = 0;
    for article in articles {
        match article {
            Ok(article) => {
                count += 1;
                if count % 10000 == 0 {
                    println!("{}", count);
                }
                //println!("{}", article);
            }
            Err(ArticleError(e)) => {
                eprintln!("Error reading article {e}");
                return;
            }
        }
    }
    println!("Read {count} articles");
}
