use std::sync::{atomic::AtomicU64, Arc};

use indicatif::{ProgressBar, ProgressStyle};
use rayon::iter::ParallelIterator;

pub trait ProgressIter<T: Send> {
    fn show_progress(self, length: u64) -> impl ParallelIterator<Item = T>;
}

impl<T: Send, I> ProgressIter<T> for I
where
    I: ParallelIterator<Item = T>,
{
    fn show_progress(self, length: u64) -> impl ParallelIterator<Item = T> {
        let progress_bar = ProgressBar::new(length);
        let counter = Arc::new(AtomicU64::new(0));

        progress_bar.set_style(
            ProgressStyle::default_bar()
                .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} ({eta})")
                .unwrap()
                .progress_chars("##-"),
        );
        self.inspect(move |_| {
            let value = counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            progress_bar.set_position(value);
        })
    }
}
