use std::io::{BufWriter, Write};

use byteorder::WriteBytesExt;

pub trait LittleEndianStruct {
    fn write_little_endian(&self, writer: &mut BufWriter<impl Write>);
}

impl LittleEndianStruct for [u64] {
    fn write_little_endian(&self, writer: &mut BufWriter<impl Write>) {
        for &num in self {
            writer.write_u64::<byteorder::LittleEndian>(num).unwrap();
        }
    }
}

impl LittleEndianStruct for [u16] {
    fn write_little_endian(&self, writer: &mut BufWriter<impl Write>) {
        for &num in self {
            writer.write_u16::<byteorder::LittleEndian>(num).unwrap();
        }
    }
}
