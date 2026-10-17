use super::*;

#[test]
fn writer_lock_releases_while_a_duplicate_descriptor_remains_open() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("session.jsonl");
    let lock = TranscriptWriterLock::acquire(&path).unwrap();
    // A forked child shares this file description until exec closes it.
    let duplicate = lock._file.try_clone().unwrap();
    assert!(TranscriptWriterLock::try_acquire(&path).unwrap().is_none());
    drop(lock);
    let next = TranscriptWriterLock::acquire(&path).unwrap();
    drop(duplicate);
    assert!(TranscriptWriterLock::try_acquire(&path).unwrap().is_none());
    drop(next);
    assert!(TranscriptWriterLock::try_acquire(&path).unwrap().is_some());
}
