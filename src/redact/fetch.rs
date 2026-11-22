use std::io::Read;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use sha2::{Digest, Sha256};

pub(crate) const REVISION: &str = "7216b6a3b897ef903d4b1ba367ae6fb95b0fa5e4";

const HOSTS: [&str; 2] = ["https://huggingface.co", "https://hf-mirror.com"];

const FILES: [(&str, u64, &str); 3] = [
    (
        "int8/model_int8.onnx",
        330_781,
        "0a070d3113615e33ba3b84d6bd486f663a377c04c4b056e7381732179f51832a",
    ),
    (
        "int8/model_int8.onnx_data",
        37_124_608,
        "0ba34216ceead01e0ae2dfd8e46d52b67e7aee80b04f722d9b758613a91c0067",
    ),
    (
        "int8/tokenizer.json",
        1_502_881,
        "0cf3b235ef125658015a6c8026a5fe6f28088bf825101a990ec18a10200feef4",
    ),
];

pub(crate) const NAMES: [&str; 3] = ["model_int8.onnx", "model_int8.onnx_data", "tokenizer.json"];

pub(crate) fn present(dir: &Path) -> bool {
    NAMES.iter().all(|name| dir.join(name).is_file())
}

pub(crate) fn fetch_missing(dir: &Path) -> Result<Vec<String>> {
    std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    let agent = ureq_like_agent();
    let mut written = Vec::new();
    for (name, size, digest) in FILES {
        let target = dir.join(Path::new(name).file_name().context("a file name")?);
        if target.is_file() && digest_of(&target)? == digest {
            continue;
        }
        let mut last: Option<anyhow::Error> = None;
        for host in HOSTS {
            let url = format!("{host}/QuantumNous/astrlink-guard/resolve/{REVISION}/{name}");
            match download(&agent, &url, &target, size, digest) {
                Ok(()) => {
                    written.push(target.display().to_string());
                    last = None;
                    break;
                }
                Err(error) => last = Some(error),
            }
        }
        if let Some(error) = last {
            return Err(error).with_context(|| format!("fetching {name}"));
        }
    }
    Ok(written)
}

fn ureq_like_agent() -> reqwest::blocking::Client {
    reqwest::blocking::Client::builder()
        .user_agent(concat!("letcode/", env!("CARGO_PKG_VERSION")))
        .timeout(std::time::Duration::from_secs(120))
        .build()
        .unwrap_or_else(|_| reqwest::blocking::Client::new())
}

fn download(
    agent: &reqwest::blocking::Client,
    url: &str,
    target: &Path,
    size: u64,
    digest: &str,
) -> Result<()> {
    let response = agent.get(url).send().context("sending the request")?;
    if !response.status().is_success() {
        bail!("{url} answered {}", response.status());
    }
    let mut hasher = Sha256::new();
    let part = part_path(target);
    let mut file =
        std::fs::File::create(&part).with_context(|| format!("writing {}", part.display()))?;
    let mut reader = response.take(size + 1);
    let mut buffer = vec![0u8; 1 << 16];
    let mut written = 0u64;
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        written += read as u64;
        if written > size {
            let _ = std::fs::remove_file(&part);
            bail!("{url} returned more than the {size} bytes of {target:?}");
        }
        hasher.update(&buffer[..read]);
        std::io::Write::write_all(&mut file, &buffer[..read])?;
    }
    drop(file);
    let actual = format!("{:x}", hasher.finalize());
    if written != size || actual != digest {
        let _ = std::fs::remove_file(&part);
        bail!("{url} returned {written} bytes with sha256 {actual}, expected {size} and {digest}");
    }
    std::fs::rename(&part, target).with_context(|| format!("moving into {}", target.display()))?;
    Ok(())
}

fn part_path(target: &Path) -> PathBuf {
    let mut name = target.as_os_str().to_owned();
    name.push(".part");
    PathBuf::from(name)
}

fn digest_of(path: &Path) -> Result<String> {
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 1 << 16];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}
