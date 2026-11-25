//! An interface for dealing with the kinds of parallel computations involved in
//! `halo2`. It's currently just a (very!) thin wrapper around `rayon` but may
//! be extended in the future to allow for various parallelism strategies.

#[cfg(all(
    feature = "multicore",
    target_arch = "wasm32",
    not(target_feature = "atomics")
))]
compile_error!(
    "The multicore feature flag is not supported on wasm32 architectures without atomics"
);

pub use maybe_rayon::{join, scope};

#[cfg(feature = "multicore")]
pub use maybe_rayon::current_num_threads;

#[cfg(not(feature = "multicore"))]
pub fn current_num_threads() -> usize {
    1
}
