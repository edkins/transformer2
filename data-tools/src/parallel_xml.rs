use std::io::{BufRead, BufReader, Read, Seek};

#[derive(Clone, Debug)]
pub struct FileSection {
    pub filename: String,
    pub start: u64,
    pub end: u64,
}

impl FileSection {
    pub fn buf_reader(&self) -> impl BufRead {
        let mut file = std::fs::File::open(&self.filename).expect("Error opening file");
        file.seek(std::io::SeekFrom::Start(self.start)).expect("Error seeking file");
        BufReader::new(file.take(self.end - self.start))
    }
}

const SKIP_SIZE:u64 = 512 * 1024 * 1024;
const BUFFER_SIZE:usize = 1024 * 1024;
const NEEDLE: &[u8] = b"\n  <page>\n";

pub fn find_page_starts(filename: &str) -> Vec<FileSection> {
    let mut page_starts = Vec::new();
    let mut buf = vec![0; BUFFER_SIZE as usize];
    let mut main_offset = 0;
    let mut file = std::fs::File::open(filename).expect("Error opening file");
    loop {
        let mut offset: u64 = main_offset;
        loop {
            file.seek(std::io::SeekFrom::Start(offset)).expect("Error seeking file");
            let bytes_read = file.read(&mut buf).expect("Error reading file");
            if bytes_read == 0 {
                return page_starts.windows(2).map(|window| {
                    FileSection {
                        filename: filename.to_string(),
                        start: window[0],
                        end: window[1],
                    }
                }).collect();
            }

            if let Some(finding) = buf[..bytes_read].windows(NEEDLE.len()).position(|window| window == NEEDLE) {
                page_starts.push(offset + finding as u64 + 1);
                break;
            }

            offset += buf.len() as u64;
        }
        main_offset += SKIP_SIZE;
    }
}
