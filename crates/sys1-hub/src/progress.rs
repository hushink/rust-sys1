//! Download progress reported while pulling a model.

/// One progress sample for a file being materialized into the cache.
#[derive(Clone, Copy, Debug)]
pub struct PullProgress<'a> {
    /// Manifest path of the file, e.g. `model-00001-of-00002.safetensors`.
    pub file: &'a str,
    /// One-based index of the file in the manifest.
    pub file_index: usize,
    /// Number of files in the manifest.
    pub file_count: usize,
    /// Bytes of this file written so far, including a resumed prefix.
    pub file_downloaded: u64,
    /// Expected size of this file.
    pub file_total: u64,
    /// Bytes of the whole model written so far, including completed files.
    pub total_downloaded: u64,
    /// Sum of every file's expected size.
    pub total_bytes: u64,
}

/// Receives download progress. `Hub::pull_with_observer` calls it from the
/// pulling thread, at most once per MiB and once per file boundary.
pub trait PullObserver {
    fn on_progress(&self, progress: &PullProgress<'_>);
}

impl<F> PullObserver for F
where
    F: Fn(&PullProgress<'_>),
{
    fn on_progress(&self, progress: &PullProgress<'_>) {
        self(progress);
    }
}

/// Adapter the hub installs for the file currently being materialized.
///
/// It is an implementation detail of [`crate::SourceContext`]; sources call
/// [`FileProgress::report`] with the bytes written so far.
pub struct FileProgress<'a> {
    observer: &'a dyn PullObserver,
    file: &'a str,
    file_index: usize,
    file_count: usize,
    file_total: u64,
    total_before: u64,
    total_bytes: u64,
}

impl<'a> FileProgress<'a> {
    pub(crate) fn new(
        observer: &'a dyn PullObserver,
        file: &'a str,
        file_index: usize,
        file_count: usize,
        file_total: u64,
        total_before: u64,
        total_bytes: u64,
    ) -> Self {
        Self {
            observer,
            file,
            file_index,
            file_count,
            file_total,
            total_before,
            total_bytes,
        }
    }

    pub fn report(&self, downloaded: u64) {
        let downloaded = downloaded.min(self.file_total);
        self.observer.on_progress(&PullProgress {
            file: self.file,
            file_index: self.file_index,
            file_count: self.file_count,
            file_downloaded: downloaded,
            file_total: self.file_total,
            total_downloaded: self.total_before + downloaded,
            total_bytes: self.total_bytes,
        });
    }
}
