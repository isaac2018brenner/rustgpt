//! Minimal data parallelism on top of `std::thread::scope`.

use std::sync::OnceLock;

pub fn num_threads() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("RUSTGPT_THREADS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, |n| n.get()))
            .max(1)
    })
}

/// Runs `f` on every item, spreading the items round-robin across worker threads.
/// Items are typically disjoint `&mut` chunks of an output buffer, so no locking is needed.
pub fn par_each<T: Send, F: Fn(T) + Sync>(items: Vec<T>, f: F) {
    let n = num_threads().min(items.len());
    if n <= 1 {
        items.into_iter().for_each(f);
        return;
    }
    let mut buckets: Vec<Vec<T>> = (0..n).map(|_| Vec::new()).collect();
    for (i, item) in items.into_iter().enumerate() {
        buckets[i % n].push(item);
    }
    let f = &f;
    std::thread::scope(|s| {
        for bucket in buckets {
            s.spawn(move || bucket.into_iter().for_each(f));
        }
    });
}
