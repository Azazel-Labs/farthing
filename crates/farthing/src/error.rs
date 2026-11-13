use thiserror::Error;

/// Errors shared across farthing crates.
#[derive(Debug, Error)]
pub enum Error {
    #[error("invalid mesh: {0}")]
    InvalidMesh(String),
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("render host error: {0}")]
    Host(String),
    #[error("the render host does not support `{0}`")]
    Unsupported(&'static str),
    #[error("image error: {0}")]
    Image(String),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T, E = Error> = std::result::Result<T, E>;
