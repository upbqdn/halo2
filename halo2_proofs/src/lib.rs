//! # halo2_proofs

#![cfg_attr(docsrs, feature(doc_cfg))]
// The actual lints we want to disable.
#![allow(clippy::op_ref, clippy::many_single_char_names)]
#![deny(rustdoc::broken_intra_doc_links)]
#![deny(missing_debug_implementations)]
#![deny(missing_docs)]
#![deny(unsafe_code)]

pub mod arithmetic;
pub mod circuit;
pub use pasta_curves as pasta;
mod multicore;
pub mod plonk;
pub mod poly;
pub mod transcript;

pub mod dev;
mod helpers;

// Copyright (c) 2026 Zakura Contributors. Licensed under MIT OR Apache-2.0.
// Ported from common/halo2_proofs/src/lib.rs at 639b159e6bf904e8 (#80), and
// common/crates/halo2_proofs/src/lib.rs at f2c0f32272fbf9ae (#503).
// Signed width nine needs one more cached multiple than unsigned width eight.
#[cfg(feature = "batch")]
const INSTANCE_WINDOW_BITS: usize = 9;
#[cfg(feature = "batch")]
const INSTANCE_WINDOW_ENTRIES_PER_BASE: usize = 1 << (INSTANCE_WINDOW_BITS - 1);
// Each cached row retains 256 affine points, so this bounds the Pasta cache at
// one MiB while covering Orchard's ten-row instance columns.
#[cfg(feature = "batch")]
const MAX_CACHED_INSTANCE_ROWS: usize = 64;
