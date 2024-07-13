use std::sync::{atomic::AtomicU64, Arc};

use indicatif::{ProgressBar, ProgressStyle};
use rayon::iter::ParallelIterator;

pub trait ParallelProgressIter<T: Send> {
    fn show_progress(self, length: u64) -> impl ParallelIterator<Item = T>;
}

impl<T: Send, I> ParallelProgressIter<T> for I
where
    I: ParallelIterator<Item = T>,
{
    fn show_progress(self, length: u64) -> impl ParallelIterator<Item = T> {
        let progress_bar = bar(length);
        let counter = Arc::new(AtomicU64::new(0));

        self.inspect(move |_| {
            let value = counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            progress_bar.set_position(value);
        })
    }
}

// pub trait ProgressIter<T> {
//     fn show_progress(self, length: u64) -> impl Iterator<Item = T>;
// }

// impl<T, I> ProgressIter<T> for I
// where
//     I: Iterator<Item = T>
// {
//     fn show_progress(self, length: u64) -> impl Iterator<Item = T> {
//         let progress_bar = bar(length);
//         self.inspect(move |_| {
//             progress_bar.inc(1);
//         })
//     }
// }

fn bar(size: u64) -> ProgressBar {
    let progress_bar = ProgressBar::new(size);
    progress_bar.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} ({eta})")
            .unwrap()
            .progress_chars("##-"),
    );
    progress_bar
}
