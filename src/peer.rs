use anyhow::{Context, Result, anyhow, bail};
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tempfile::TempDir;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWriteExt};
use tokio::task::{JoinHandle, JoinSet};
use tokio::time::timeout;

const MAX_MESSAGE_BYTES: usize = 1024 * 1024;
const IO_TIMEOUT: Duration = Duration::from_secs(5);
const MAX_CONNECTIONS: usize = 16;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct PeerAddress {
    pub instance_id: String,
    pub session_id: String,
    pub generation: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PeerInfo {
    pub address: PeerAddress,
    pub endpoint: String,
    pub workspace: Option<PathBuf>,
    pub title: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PeerMessage {
    pub from: PeerAddress,
    pub to: PeerAddress,
    pub message: String,
}

#[derive(Clone)]
pub struct PeerClient {
    peers_dir: PathBuf,
    info: Arc<Mutex<PeerInfo>>,
}

impl PeerClient {
    pub fn info(&self) -> PeerInfo {
        self.info.lock().unwrap().clone()
    }

    pub fn discover(&self) -> Result<Vec<PeerInfo>> {
        let entries = match fs::read_dir(&self.peers_dir) {
            Ok(entries) => entries,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(error) => return Err(error.into()),
        };
        let mut peers = Vec::new();
        for entry in entries {
            let path = entry?.path();
            if path.extension().is_none_or(|extension| extension != "json") {
                continue;
            }
            let bytes = match fs::read(&path) {
                Ok(bytes) => bytes,
                Err(error) if error.kind() == io::ErrorKind::NotFound => continue,
                Err(error) => return Err(error.into()),
            };
            match serde_json::from_slice(&bytes) {
                Ok(peer) => peers.push(peer),
                Err(error) if error.is_eof() || error.is_syntax() || error.is_data() => continue,
                Err(error) => return Err(error.into()),
            }
        }
        Ok(peers)
    }

    pub async fn send(&self, to: PeerAddress, message: String) -> Result<()> {
        let peer = self
            .discover()?
            .into_iter()
            .find(|peer| peer.address.instance_id == to.instance_id)
            .ok_or_else(|| anyhow!("peer instance not found: {}", to.instance_id))?;
        let bytes = serde_json::to_vec(&PeerMessage {
            from: self.info().address,
            to,
            message,
        })?;
        if bytes.len() > MAX_MESSAGE_BYTES {
            bail!("peer message exceeds {MAX_MESSAGE_BYTES} bytes");
        }
        timeout(IO_TIMEOUT, async {
            let mut stream = transport::connect(&peer.endpoint).await?;
            stream.write_u32(bytes.len() as u32).await?;
            stream.write_all(&bytes).await?;
            stream.shutdown().await?;
            Ok::<_, anyhow::Error>(())
        })
        .await
        .context("peer send timed out")?
    }
}

pub struct PeerRuntime {
    client: PeerClient,
    listener: JoinHandle<()>,
    _endpoint_dir: TempDir,
}

impl PeerRuntime {
    pub async fn start(
        config_dir: &Path,
        workspace: Option<PathBuf>,
        session_id: String,
        title: Option<String>,
        on_message: impl Fn(PeerMessage) + Send + Sync + 'static,
    ) -> Result<Self> {
        let peers_dir = config_dir.join("peers");
        fs::create_dir_all(&peers_dir)?;
        let endpoint_dir = transport::endpoint_dir()?;
        let instance_id = endpoint_dir
            .path()
            .file_name()
            .unwrap()
            .to_string_lossy()
            .into_owned();
        let endpoint = transport::endpoint(&endpoint_dir);
        let listener = transport::Listener::bind(&endpoint)?;
        let info = PeerInfo {
            address: PeerAddress {
                instance_id,
                session_id,
                generation: 0,
            },
            endpoint,
            workspace,
            title,
        };
        write_registration(&peers_dir, &info)?;
        let client = PeerClient {
            peers_dir,
            info: Arc::new(Mutex::new(info)),
        };
        Ok(Self {
            client,
            listener: tokio::spawn(listen(listener, Arc::new(on_message))),
            _endpoint_dir: endpoint_dir,
        })
    }

    pub fn client(&self) -> PeerClient {
        self.client.clone()
    }

    pub fn update_session(&self, session_id: String, title: Option<String>) -> Result<()> {
        self.set_session(session_id, title, false)
    }

    pub fn resume_session(&self, session_id: String, title: Option<String>) -> Result<()> {
        self.set_session(session_id, title, true)
    }

    fn set_session(&self, session_id: String, title: Option<String>, resumed: bool) -> Result<()> {
        let mut current = self.client.info.lock().unwrap();
        let mut next = current.clone();
        if resumed || next.address.session_id != session_id {
            next.address.generation += 1;
        }
        next.address.session_id = session_id;
        next.title = title;
        write_registration(&self.client.peers_dir, &next)?;
        *current = next;
        Ok(())
    }
}

impl Drop for PeerRuntime {
    fn drop(&mut self) {
        self.listener.abort();
        let path = registration_path(
            &self.client.peers_dir,
            &self.client.info().address.instance_id,
        );
        if let Err(error) = fs::remove_file(&path)
            && error.kind() != io::ErrorKind::NotFound
        {
            tracing::warn!(%error, path = %path.display(), "peer registration cleanup failed");
        }
    }
}

fn registration_path(peers_dir: &Path, instance_id: &str) -> PathBuf {
    peers_dir.join(format!("{instance_id}.json"))
}

fn write_registration(peers_dir: &Path, info: &PeerInfo) -> Result<()> {
    let mut file = tempfile::NamedTempFile::new_in(peers_dir)?;
    serde_json::to_writer(&mut file, info)?;
    file.flush()?;
    file.persist(registration_path(peers_dir, &info.address.instance_id))?;
    Ok(())
}

async fn read_message(mut stream: impl AsyncRead + Unpin) -> Result<PeerMessage> {
    timeout(IO_TIMEOUT, async {
        let length = stream.read_u32().await? as usize;
        if length > MAX_MESSAGE_BYTES {
            bail!("peer message exceeds {MAX_MESSAGE_BYTES} bytes");
        }
        let mut bytes = vec![0; length];
        stream.read_exact(&mut bytes).await?;
        Ok(serde_json::from_slice(&bytes)?)
    })
    .await
    .context("peer receive timed out")?
}

async fn listen(
    mut listener: transport::Listener,
    on_message: Arc<dyn Fn(PeerMessage) + Send + Sync>,
) {
    let mut connections = JoinSet::new();
    loop {
        tokio::select! {
            accepted = listener.accept(), if connections.len() < MAX_CONNECTIONS => {
                match accepted {
                    Ok(stream) => {
                        let on_message = on_message.clone();
                        connections.spawn(async move {
                            let message = read_message(stream).await?;
                            on_message(message);
                            Ok::<_, anyhow::Error>(())
                        });
                    }
                    Err(error) => {
                        tracing::warn!(%error, "peer listener failed");
                        break;
                    }
                }
            }
            completed = connections.join_next(), if !connections.is_empty() => {
                match completed {
                    Some(Ok(Ok(()))) => {}
                    Some(Ok(Err(error))) => tracing::warn!(%error, "peer receive failed"),
                    Some(Err(error)) => tracing::warn!(%error, "peer receive task failed"),
                    None => {}
                }
            }
        }
    }
}

#[cfg(unix)]
mod transport {
    use super::*;
    use tokio::net::{UnixListener, UnixStream};

    pub struct Listener(UnixListener);

    pub fn endpoint_dir() -> io::Result<TempDir> {
        tempfile::Builder::new()
            .prefix("lc-p-")
            .rand_bytes(16)
            .tempdir_in("/tmp")
    }

    pub fn endpoint(dir: &TempDir) -> String {
        dir.path().join("socket").to_string_lossy().into_owned()
    }

    impl Listener {
        pub fn bind(endpoint: &str) -> io::Result<Self> {
            UnixListener::bind(endpoint).map(Self)
        }

        pub async fn accept(&mut self) -> io::Result<UnixStream> {
            self.0.accept().await.map(|(stream, _)| stream)
        }
    }

    pub async fn connect(endpoint: &str) -> io::Result<UnixStream> {
        UnixStream::connect(endpoint).await
    }
}

#[cfg(windows)]
mod transport {
    use super::*;
    use tokio::net::windows::named_pipe::{
        ClientOptions, NamedPipeClient, NamedPipeServer, ServerOptions,
    };

    pub struct Listener {
        server: NamedPipeServer,
        endpoint: String,
    }

    pub fn endpoint_dir() -> io::Result<TempDir> {
        tempfile::Builder::new()
            .prefix("lc-p-")
            .rand_bytes(16)
            .tempdir()
    }

    pub fn endpoint(dir: &TempDir) -> String {
        format!(
            r"\\.\pipe\letcode-{}",
            dir.path().file_name().unwrap().to_string_lossy()
        )
    }

    fn server(endpoint: &str, first: bool) -> io::Result<NamedPipeServer> {
        ServerOptions::new()
            .first_pipe_instance(first)
            .reject_remote_clients(true)
            .access_outbound(false)
            .create(endpoint)
    }

    impl Listener {
        pub fn bind(endpoint: &str) -> io::Result<Self> {
            Ok(Self {
                server: server(endpoint, true)?,
                endpoint: endpoint.to_owned(),
            })
        }

        pub async fn accept(&mut self) -> io::Result<NamedPipeServer> {
            self.server.connect().await?;
            let next = server(&self.endpoint, false)?;
            Ok(std::mem::replace(&mut self.server, next))
        }
    }

    pub async fn connect(endpoint: &str) -> io::Result<NamedPipeClient> {
        ClientOptions::new().read(false).open(endpoint)
    }
}

#[cfg(test)]
#[path = "../tests/internal/peer/tests.rs"]
mod tests;
