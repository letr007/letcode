use super::restore_latest_fake_client;
use crate::fake::FakeClient;
use crate::transcript::{TranscriptRecorder, read_records};

#[test]
fn restores_fake_client_written_to_a_session_transcript() {
    let directory = tempfile::tempdir().expect("create transcript directory");
    let mut recorder = TranscriptRecorder::create(directory.path()).expect("create recorder");
    recorder
        .record_fake_client_changed(None, Some(FakeClient::Codex))
        .expect("record fake client change");
    let transcript_path = recorder.path().to_path_buf();
    drop(recorder);

    let records = read_records(&transcript_path).expect("read session transcript");

    assert_eq!(
        restore_latest_fake_client(&records),
        Some(FakeClient::Codex)
    );
}

#[test]
fn fake_client_state_is_scoped_to_each_session_transcript() {
    let directory = tempfile::tempdir().expect("create transcript directory");

    let mut first_recorder =
        TranscriptRecorder::create(directory.path()).expect("create first recorder");
    first_recorder
        .record_fake_client_changed(None, Some(FakeClient::Codex))
        .expect("record first fake client change");
    let first_transcript_path = first_recorder.path().to_path_buf();
    drop(first_recorder);

    let mut second_recorder =
        TranscriptRecorder::create(directory.path()).expect("create second recorder");
    second_recorder
        .record_fake_client_changed(None, Some(FakeClient::Anthropic))
        .expect("record second fake client change");
    let second_transcript_path = second_recorder.path().to_path_buf();
    drop(second_recorder);

    let first_records = read_records(&first_transcript_path).expect("read first transcript");
    let second_records = read_records(&second_transcript_path).expect("read second transcript");

    assert_eq!(
        restore_latest_fake_client(&first_records),
        Some(FakeClient::Codex)
    );
    assert_eq!(
        restore_latest_fake_client(&second_records),
        Some(FakeClient::Anthropic)
    );
}
