use super::*;

#[test]
fn replacement_failure_keeps_the_source_and_original_file_names() {
    for error_code in [5, 1175, 1176] {
        let directory = tempfile::tempdir().unwrap();
        let source = directory.path().join("new.toml");
        let destination = directory.path().join("config.toml");
        fs::write(&source, "new").unwrap();
        fs::write(&destination, "old").unwrap();
        let mut backup = None;
        let error = replace_file_with_backup(
            &source,
            &destination,
            |_, _, path| {
                backup = Some(path.to_path_buf());
                Err(std::io::Error::from_raw_os_error(error_code))
            },
            |_| panic!("failed replacements must not be cleaned up"),
        )
        .unwrap_err();
        let backup = backup.unwrap();
        assert_eq!(fs::read_to_string(&source).unwrap(), "new");
        assert_eq!(fs::read_to_string(&destination).unwrap(), "old");
        assert!(backup.parent().unwrap().is_dir());
        assert!(!backup.exists());
        assert_eq!(
            error
                .downcast_ref::<std::io::Error>()
                .unwrap()
                .raw_os_error(),
            Some(error_code)
        );
        let detail = format!("{error:#}");
        assert!(detail.contains(source.to_str().unwrap()));
        assert!(detail.contains(destination.to_str().unwrap()));
        assert!(detail.contains(backup.parent().unwrap().to_str().unwrap()));
    }
}

#[test]
fn partial_replacement_failure_keeps_both_recovery_files() {
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("new.toml");
    let destination = directory.path().join("config.toml");
    fs::write(&source, "new").unwrap();
    fs::write(&destination, "old").unwrap();
    let mut backup = None;
    let error = replace_file_with_backup(
        &source,
        &destination,
        |_, destination, path| {
            backup = Some(path.to_path_buf());
            fs::rename(destination, path)?;
            Err(std::io::Error::from_raw_os_error(1177))
        },
        |_| panic!("partial replacements must not be cleaned up"),
    )
    .unwrap_err();
    let backup = backup.unwrap();
    assert!(!destination.exists());
    assert_eq!(fs::read_to_string(&source).unwrap(), "new");
    assert_eq!(fs::read_to_string(&backup).unwrap(), "old");
    assert_eq!(
        error
            .downcast_ref::<std::io::Error>()
            .unwrap()
            .raw_os_error(),
        Some(1177)
    );
    assert!(format!("{error:#}").contains(backup.parent().unwrap().to_str().unwrap()));
}

#[test]
fn successful_replacement_cleans_up_its_backup() {
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("new.toml");
    let destination = directory.path().join("config.toml");
    fs::write(&source, "new").unwrap();
    fs::write(&destination, "old").unwrap();
    let mut backup = None;
    replace_file_with_backup(
        &source,
        &destination,
        |source, destination, path| {
            backup = Some(path.to_path_buf());
            fs::rename(destination, path)?;
            fs::rename(source, destination)
        },
        |path| fs::remove_dir_all(path),
    )
    .unwrap();
    assert_eq!(fs::read_to_string(&destination).unwrap(), "new");
    assert!(!source.exists());
    assert!(!backup.unwrap().parent().unwrap().exists());
}

#[test]
fn backup_cleanup_failure_does_not_report_a_successful_replace_as_failed() {
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("new.toml");
    let destination = directory.path().join("config.toml");
    fs::write(&source, "new").unwrap();
    fs::write(&destination, "old").unwrap();
    let mut backup = None;
    replace_file_with_backup(
        &source,
        &destination,
        |source, destination, path| {
            backup = Some(path.to_path_buf());
            fs::rename(destination, path)?;
            fs::rename(source, destination)
        },
        |_| Err(std::io::Error::from(std::io::ErrorKind::PermissionDenied)),
    )
    .unwrap();
    assert_eq!(fs::read_to_string(&destination).unwrap(), "new");
    assert_eq!(fs::read_to_string(backup.unwrap()).unwrap(), "old");
    assert!(!source.exists());
}

#[test]
fn config_preparation_failure_removes_only_the_new_temporary_file() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("config.toml");
    fs::write(&path, "original").unwrap();
    let metadata = fs::metadata(&path).unwrap();
    fs::write(&path, "external edit").unwrap();
    atomic_write_config(&path, "original", &metadata, b"new").unwrap_err();
    assert_eq!(fs::read_to_string(&path).unwrap(), "external edit");
    assert_eq!(fs::read_dir(directory.path()).unwrap().count(), 1);
}
