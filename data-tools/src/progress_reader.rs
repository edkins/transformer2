use indicatif::{ProgressBar, ProgressStyle};
use std::io::{self, Read};

pub struct ProgressReader<R: Read> {
    inner: R,
    progress_bar: ProgressBar,
    current: u64,
}

impl<R: Read> ProgressReader<R> {
    pub fn new(inner: R, total_size: u64) -> Self {
        let progress_bar = ProgressBar::new(total_size);
        progress_bar.set_style(
            ProgressStyle::default_bar()
                .template("[{elapsed_precise}] {bar:40.cyan/blue} {bytes}/{total_bytes} ({eta})")
                .unwrap()
                .progress_chars("##-"),
        );

        ProgressReader {
            inner,
            progress_bar,
            current: 0,
        }
    }
}

impl<R: Read> Read for ProgressReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let result = self.inner.read(buf)?;
        self.current += result as u64;
        self.progress_bar.set_position(self.current);
        Ok(result)
    }
}
