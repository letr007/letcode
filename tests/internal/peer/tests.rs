use super::*;

#[tokio::test]
async fn registration_tracks_session_and_runtime_lifetime() -> Result<()> {
    let config = tempfile::tempdir()?;
    let runtime = PeerRuntime::start(
        config.path(),
        Some(PathBuf::from("workspace")),
        "first".into(),
        None,
        |_| {},
    )
    .await?;
    let client = runtime.client();
    let initial = client.info();
    let path = registration_path(&client.peers_dir, &initial.address.instance_id);
    assert!(path.is_file());
    assert_eq!(initial.workspace, Some(PathBuf::from("workspace")));

    runtime.update_session("first".into(), Some("title".into()))?;
    assert_eq!(client.info().address, initial.address);
    assert_eq!(client.discover()?[0].title.as_deref(), Some("title"));

    runtime.update_session("second".into(), None)?;
    assert_eq!(client.info().address.session_id, "second");
    assert_eq!(client.info().address.generation, 1);
    runtime.resume_session("second".into(), None)?;
    assert_eq!(client.info().address.generation, 2);
    assert_eq!(client.discover()?[0].address, client.info().address);

    let endpoint_dir = runtime._endpoint_dir.path().to_owned();
    drop(runtime);
    assert!(!path.exists());
    assert!(!endpoint_dir.exists());
    assert!(client.discover()?.is_empty());
    Ok(())
}

#[cfg(unix)]
#[tokio::test]
async fn short_connections_exchange_messages_while_another_connection_is_incomplete() -> Result<()>
{
    let config = tempfile::tempdir()?;
    let (first_tx, mut first_rx) = tokio::sync::mpsc::unbounded_channel();
    let first = PeerRuntime::start(config.path(), None, "first".into(), None, move |message| {
        first_tx.send(message).unwrap();
    })
    .await?;
    let (second_tx, mut second_rx) = tokio::sync::mpsc::unbounded_channel();
    let second = PeerRuntime::start(config.path(), None, "second".into(), None, move |message| {
        second_tx.send(message).unwrap();
    })
    .await?;
    let first_client = first.client();
    let second_client = second.client();
    let incomplete = tokio::net::UnixStream::connect(&second_client.info().endpoint).await?;

    first_client
        .send(second_client.info().address, "hello".into())
        .await?;
    let received = timeout(Duration::from_secs(1), second_rx.recv())
        .await?
        .unwrap();
    assert_eq!(received.from, first_client.info().address);
    assert_eq!(received.to, second_client.info().address);
    assert_eq!(received.message, "hello");

    second_client.send(received.from, "reply".into()).await?;
    let received = timeout(Duration::from_secs(1), first_rx.recv())
        .await?
        .unwrap();
    assert_eq!(received.from, second_client.info().address);
    assert_eq!(received.message, "reply");
    drop(incomplete);
    Ok(())
}

#[tokio::test]
async fn discovery_skips_incomplete_records_and_reports_io_errors() -> Result<()> {
    let config = tempfile::tempdir()?;
    let runtime = PeerRuntime::start(config.path(), None, "first".into(), None, |_| {}).await?;
    let client = runtime.client();
    fs::write(client.peers_dir.join("incomplete.json"), b"{")?;
    fs::write(client.peers_dir.join("missing-fields.json"), b"{}")?;
    assert_eq!(client.discover()?.len(), 1);
    fs::create_dir(client.peers_dir.join("unreadable.json"))?;
    assert!(client.discover().is_err());
    Ok(())
}

#[cfg(unix)]
#[tokio::test]
async fn send_reports_missing_and_disconnected_endpoints() -> Result<()> {
    let config = tempfile::tempdir()?;
    let runtime = PeerRuntime::start(config.path(), None, "first".into(), None, |_| {}).await?;
    let destination =
        PeerRuntime::start(config.path(), None, "second".into(), None, |_| {}).await?;
    let client = runtime.client();
    let mut stale = destination.client().info();
    drop(destination);
    assert!(
        client
            .send(stale.address.clone(), "missing".into())
            .await
            .is_err()
    );

    stale.endpoint = config
        .path()
        .join("absent.socket")
        .to_string_lossy()
        .into_owned();
    write_registration(&client.peers_dir, &stale)?;
    assert!(
        client
            .send(stale.address, "disconnected".into())
            .await
            .is_err()
    );
    Ok(())
}
