use std::io::{Read, Seek};

pub fn find_bz2_streams(filename: &str) -> Vec<u64> {
    let file = std::fs::File::open(filename).unwrap();
    let mut reader = std::io::BufReader::new(file);
    let mut streams = Vec::new();
    let mut offset = 0;
    loop {
        let mut stream = [0; 4];
        if reader.read_exact(&mut stream).is_err() {
            break;
        }
        if &stream[0..3] != b"BZh" {
            panic!("Not a bzip2 stream");
        }
        println!("Stream {}", streams.len());
        streams.push(offset);
        offset += 4;

        let mut block_num = 0;
        loop {
            let mut block = [0; 6];
            if reader.read_exact(&mut block).is_err() {
                panic!("Unexpected end of file");
            }
            offset += 6;
            
            println!("  Block {} {:?}", block_num, block);
            block_num += 1;
            match block {
                [0x17, 0x72, 0x45, 0x38, 0x50, 0x90] => break, // end of stream marker
                [0x31, 0x41, 0x59, 0x26, 0x53, 0x59] => {
                    // block header
                    let mut crc_size = [0; 8];
                    if reader.read_exact(&mut crc_size).is_err() {
                        panic!("Unexpected end of file");
                    }
                    let block_size = u32::from_be_bytes([0, crc_size[5], crc_size[6], crc_size[7]]);
                    println!("    Block size: {}", block_size);
                    offset += (block_size-1) as u64;
                },
                _ => panic!("Invalid block header"),
            }
        }
        reader.seek(std::io::SeekFrom::Start(offset)).unwrap();
    }
    streams
}