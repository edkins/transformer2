use std::io::{BufRead, BufReader, Read, Seek};

use byteorder::ReadBytesExt;

use crate::progress_reader::ProgressReader;

#[derive(Clone, Debug)]
pub struct FileSection {
    pub filename: String,
    pub start: u64,
    pub end: u64,
}

impl FileSection {
    pub fn reader(&self) -> impl Read {
        let mut file = std::fs::File::open(&self.filename).expect("Error opening file");
        file.seek(std::io::SeekFrom::Start(self.start))
            .expect("Error seeking file");
        file.take(self.end - self.start)
    }

    pub fn buf_reader(&self) -> impl BufRead {
        BufReader::new(self.reader())
    }
}

pub fn find_all_page_starts(filename: &str) -> Vec<u64> {
    let mut page_starts = Vec::new();
    let size = std::fs::metadata(filename)
        .expect("Error getting file metadata")
        .len();
    let file = std::fs::File::open(filename).expect("Error opening file");
    let progress_reader = ProgressReader::new(file, size);
    let buf_reader = BufReader::new(progress_reader);
    let mut pos = 0;
    for line in buf_reader.lines() {
        let line = line.expect("Error reading line");
        if line == "  <page>" {
            page_starts.push(pos);
        }
        pos += line.len() as u64 + 1;
    }
    pos -= 1; // last line doesn't end in newline
    assert_eq!(pos, size);
    page_starts.push(pos);
    page_starts
}

pub fn read_page_starts(filename: &str, split_filename: &str) -> Vec<FileSection> {
    let split_file = std::fs::File::open(split_filename).expect("Error opening file");
    let mut split_reader = BufReader::new(split_file);
    let mut splits = Vec::new();
    loop {
        if let Ok(word) = split_reader.read_u64::<byteorder::LittleEndian>() {
            splits.push(word);
        } else {
            break;
        }
    }
    println!("{} articles", splits.len() - 1);
    splits
        .windows(2)
        .map(|window| FileSection {
            filename: filename.to_owned(),
            start: window[0],
            end: window[1],
        })
        .collect()
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