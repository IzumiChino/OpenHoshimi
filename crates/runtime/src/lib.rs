//! Shared runtime plumbing for OpenHoshimi decode pipelines.

#![deny(missing_docs)]
#![forbid(unsafe_code)]

/// Analytic signal source (Hilbert transform wrapper).
pub mod analytic;

/// Shared decode pipeline and runtime state.
pub mod pipeline;
