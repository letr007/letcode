use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, bail};

pub(crate) struct Tensor {
    pub(crate) dtype: i32,
    values: Values,
}

enum Values {
    Raw(Vec<u8>),
    Floats(Vec<f32>),
    Ints(Vec<i64>),
}

pub(crate) struct Node {
    pub(crate) op: String,
    pub(crate) name: String,
    pub(crate) inputs: Vec<String>,
}

#[derive(Default)]
pub(crate) struct Graph {
    tensors: HashMap<String, Tensor>,
    pub(crate) nodes: Vec<Node>,
}

impl Graph {
    pub(crate) fn f32(&self, name: &str) -> Result<Vec<f32>> {
        let t = self.tensor(name)?;
        if t.dtype != 1 {
            bail!("{name} holds type {}, not float32", t.dtype);
        }
        Ok(match &t.values {
            Values::Raw(raw) => raw
                .chunks(4)
                .filter_map(|c| c.try_into().ok().map(f32::from_le_bytes))
                .collect(),
            Values::Floats(v) => v.clone(),
            Values::Ints(v) => v.iter().map(|&x| x as f32).collect(),
        })
    }

    pub(crate) fn i8(&self, name: &str) -> Result<Vec<i8>> {
        let t = self.tensor(name)?;
        if t.dtype != 3 {
            bail!("{name} holds type {}, not int8", t.dtype);
        }
        Ok(match &t.values {
            Values::Raw(raw) => raw.iter().map(|&b| b as i8).collect(),
            Values::Ints(v) => v.iter().map(|&x| x as i8).collect(),
            Values::Floats(_) => bail!("{name} holds floats, not int8"),
        })
    }

    pub(crate) fn u8(&self, name: &str) -> Result<Vec<u8>> {
        let t = self.tensor(name)?;
        if t.dtype != 2 {
            bail!("{name} holds type {}, not uint8", t.dtype);
        }
        Ok(match &t.values {
            Values::Raw(raw) => raw.clone(),
            Values::Ints(v) => v.iter().map(|&x| x as u8).collect(),
            Values::Floats(_) => bail!("{name} holds floats, not uint8"),
        })
    }

    /// Zero points are int8 beside int8 weights and uint8 beside uint8 ones.
    pub(crate) fn zp(&self, name: &str) -> Result<Vec<i32>> {
        let t = self.tensor(name)?;
        Ok(match &t.values {
            Values::Raw(raw) => match t.dtype {
                2 => raw.iter().map(|&b| i32::from(b)).collect(),
                3 => raw.iter().map(|&b| i32::from(b as i8)).collect(),
                other => bail!("{name} holds zero points of type {other}"),
            },
            Values::Ints(v) => v.iter().map(|&x| x as i32).collect(),
            Values::Floats(_) => bail!("{name} holds float zero points"),
        })
    }

    fn tensor(&self, name: &str) -> Result<&Tensor> {
        self.tensors
            .get(name)
            .with_context(|| format!("the model has no initializer {name}"))
    }
}

pub(crate) fn read(path: &Path) -> Result<Graph> {
    let data = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let dir = path.parent().unwrap_or(Path::new("."));
    let mut graph = Graph::default();
    let mut r = Reader::new(&data);
    while let Some((field, value)) = r.field()? {
        if field == 7 {
            parse_graph(value.bytes()?, dir, &mut graph)?;
        }
    }
    Ok(graph)
}

fn parse_graph(data: &[u8], dir: &Path, graph: &mut Graph) -> Result<()> {
    let mut r = Reader::new(data);
    let mut tensors = Vec::new();
    let mut nodes = Vec::new();
    while let Some((field, value)) = r.field()? {
        match field {
            1 => nodes.push(parse_node(value.bytes()?)?),
            5 => tensors.push(parse_tensor(value.bytes()?)?),
            _ => {}
        }
    }
    graph.nodes = nodes;
    let mut blobs = HashMap::new();
    for mut t in tensors {
        if let Some(external) = t.external.take() {
            let file = external
                .iter()
                .find(|(k, _)| k == "location")
                .map(|(_, v)| v.clone())
                .context("external data without a location")?;
            let offset = external
                .iter()
                .find(|(k, _)| k == "offset")
                .map(|(_, v)| v.parse::<usize>())
                .transpose()
                .context("external data offset")?
                .unwrap_or(0);
            let length = external
                .iter()
                .find(|(k, _)| k == "length")
                .map(|(_, v)| v.parse::<usize>())
                .transpose()
                .context("external data length")?;
            let path = dir.join(file);
            let blob = match blobs.entry(path.clone()) {
                std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
                std::collections::hash_map::Entry::Vacant(entry) => {
                    let blob = std::fs::read(&path)
                        .with_context(|| format!("reading external data {}", path.display()))?;
                    entry.insert(blob)
                }
            };
            let end = length.map_or(blob.len(), |l| offset + l);
            t.values = Values::Raw(
                blob.get(offset..end)
                    .context("external data out of range")?
                    .to_vec(),
            );
        }
        graph.tensors.insert(
            t.name,
            Tensor {
                dtype: t.dtype,
                values: t.values,
            },
        );
    }
    Ok(())
}

struct RawTensor {
    name: String,
    dtype: i32,
    values: Values,
    external: Option<Vec<(String, String)>>,
}

fn parse_tensor(data: &[u8]) -> Result<RawTensor> {
    let mut dims = Vec::new();
    let mut dtype = 0;
    let mut name = String::new();
    let mut raw = None;
    let mut floats = Vec::new();
    let mut ints = Vec::new();
    let mut external = None;
    let mut r = Reader::new(data);
    while let Some((field, value)) = r.field()? {
        match field {
            1 => match value {
                Value::Varint(v) => dims.push(v as usize),
                Value::Bytes(b) => dims.extend(packed_varints(b)?.into_iter().map(|v| v as usize)),
            },
            2 => dtype = value.varint()? as i32,
            4 => {
                let bytes = value.bytes()?;
                floats.extend(
                    bytes
                        .chunks(4)
                        .filter_map(|c| c.try_into().ok().map(f32::from_le_bytes)),
                );
            }
            5 | 7 => match value {
                Value::Varint(v) => ints.push(v as i64),
                Value::Bytes(b) => ints.extend(packed_varints(b)?),
            },
            8 => name = String::from_utf8(value.bytes()?.to_vec())?,
            9 => raw = Some(value.bytes()?.to_vec()),
            13 => {
                let mut entry = Reader::new(value.bytes()?);
                let (mut key, mut val) = (String::new(), String::new());
                while let Some((f, v)) = entry.field()? {
                    match f {
                        1 => key = String::from_utf8(v.bytes()?.to_vec())?,
                        2 => val = String::from_utf8(v.bytes()?.to_vec())?,
                        _ => {}
                    }
                }
                external.get_or_insert_with(Vec::new).push((key, val));
            }
            _ => {}
        }
    }
    let values = match raw {
        Some(bytes) => Values::Raw(bytes),
        None if !floats.is_empty() => Values::Floats(floats),
        None => Values::Ints(ints),
    };
    Ok(RawTensor {
        name,
        dtype,
        values,
        external,
    })
}

fn packed_varints(data: &[u8]) -> Result<Vec<i64>> {
    let mut r = Reader::new(data);
    let mut out = Vec::new();
    while r.at < data.len() {
        out.push(r.varint()? as i64);
    }
    Ok(out)
}

fn parse_node(data: &[u8]) -> Result<Node> {
    let mut op = String::new();
    let mut name = String::new();
    let mut inputs = Vec::new();
    let mut r = Reader::new(data);
    while let Some((field, value)) = r.field()? {
        match field {
            1 => inputs.push(String::from_utf8(value.bytes()?.to_vec())?),
            3 => name = String::from_utf8(value.bytes()?.to_vec())?,
            4 => op = String::from_utf8(value.bytes()?.to_vec())?,
            _ => {}
        }
    }
    Ok(Node { op, name, inputs })
}

enum Value<'a> {
    Varint(u64),
    Bytes(&'a [u8]),
}

impl<'a> Value<'a> {
    fn bytes(&self) -> Result<&'a [u8]> {
        match self {
            Value::Bytes(b) => Ok(b),
            Value::Varint(_) => bail!("a length-delimited field was a number"),
        }
    }

    fn varint(&self) -> Result<u64> {
        match self {
            Value::Varint(v) => Ok(*v),
            Value::Bytes(_) => bail!("a number field was length-delimited"),
        }
    }
}

struct Reader<'a> {
    data: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, at: 0 }
    }

    fn varint(&mut self) -> Result<u64> {
        let mut value = 0u64;
        let mut shift = 0;
        loop {
            let byte = *self
                .data
                .get(self.at)
                .context("a varint runs past the data")?;
            self.at += 1;
            value |= u64::from(byte & 0x7f) << shift;
            if byte & 0x80 == 0 {
                return Ok(value);
            }
            shift += 7;
            if shift >= 64 {
                bail!("a varint is too long");
            }
        }
    }

    fn field(&mut self) -> Result<Option<(u64, Value<'a>)>> {
        if self.at >= self.data.len() {
            return Ok(None);
        }
        let key = self.varint()?;
        let field = key >> 3;
        match key & 7 {
            0 => Ok(Some((field, Value::Varint(self.varint()?)))),
            2 => {
                let len = self.varint()? as usize;
                let start = self.at;
                self.at += len;
                let slice = self
                    .data
                    .get(start..self.at)
                    .context("a field runs past the data")?;
                Ok(Some((field, Value::Bytes(slice))))
            }
            1 => {
                let start = self.at;
                self.at += 8;
                let slice = self
                    .data
                    .get(start..self.at)
                    .context("a fixed64 field runs past the data")?;
                Ok(Some((field, Value::Bytes(slice))))
            }
            5 => {
                let start = self.at;
                self.at += 4;
                let slice = self
                    .data
                    .get(start..self.at)
                    .context("a fixed32 field runs past the data")?;
                Ok(Some((field, Value::Bytes(slice))))
            }
            other => bail!("protobuf wire type {other} is not used by ONNX"),
        }
    }
}
