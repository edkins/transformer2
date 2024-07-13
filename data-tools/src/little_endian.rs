use std::io::Write;

use byteorder::WriteBytesExt;

pub trait LittleEndianStruct {
    fn write_little_endian(&self, writer: impl Write);
}

impl LittleEndianStruct for [u64] {
    fn write_little_endian(&self, writer: impl Write) {
        let mut out_buf = std::io::BufWriter::new(writer);
        for &num in self {
            out_buf.write_u64::<byteorder::LittleEndian>(num).unwrap();
        }
    }
}

impl LittleEndianStruct for [u16] {
    fn write_little_endian(&self, writer: impl Write) {
        let mut out_buf = std::io::BufWriter::new(writer);
        for &num in self {
            out_buf.write_u16::<byteorder::LittleEndian>(num).unwrap();
        }
    }
}
