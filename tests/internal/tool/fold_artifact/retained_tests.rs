use super::*;
#[test]
fn retained_body_survives_removal_of_temporary_original() {
    let root = tempfile::tempdir().unwrap();
    let source = root.path().join("temporary.txt");
    std::fs::write(&source, "complete output").unwrap();
    let path = persist_artifact(&source, &root.path().join("session")).unwrap();
    assert_eq!(
        path,
        persist_artifact(&source, &root.path().join("session")).unwrap()
    );
    std::fs::remove_file(source).unwrap();
    assert_eq!(std::fs::read_to_string(path).unwrap(), "complete output");
}
