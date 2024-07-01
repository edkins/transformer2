use quick_xml::events::Event;
use quick_xml::reader::Reader;
use std::io::BufRead;

fn main() {
    let filename = "../enwiki-latest-pages-articles-multistream.xml.bz2";
    let file = std::fs::File::open(filename).unwrap();
    let decomp = bzip2::read::MultiBzDecoder::new(file);
    let bdecomp = std::io::BufReader::new(decomp);
    parse_xml(bdecomp).unwrap();
}

fn parse_xml<R: BufRead>(reader: R) -> Result<(), Box<dyn std::error::Error>> {
    let mut xml_reader = Reader::from_reader(reader);
    let mut buf = Vec::new();

    loop {
        match xml_reader.read_event_into(&mut buf)? {
            Event::Start(e) => println!("Start: {:?}", e.name()),
            Event::End(e) => println!("End: {:?}", e.name()),
            Event::Text(e) => println!("Text: {:?}", e.unescape()?),
            Event::Eof => break,
            _ => (), // There are other event types we're not handling here
        }

        buf.clear();
    }

    Ok(())
}
