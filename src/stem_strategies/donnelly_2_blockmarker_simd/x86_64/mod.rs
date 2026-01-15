//! x86_64 SIMD implementations for Donnelly block marker strategy

// Import architecture-specific implementations
#[cfg(target_feature = "avx2")]
pub mod avx2;

#[cfg(target_feature = "avx512f")]
pub mod avx512;

// Re-export comparison functions based on available features
#[cfg(target_feature = "avx512f")]
pub use avx512::{compare_block3_f64_avx512, compare_block4_f32_avx512};

#[cfg(all(target_feature = "avx2", not(target_feature = "avx512f")))]
pub use avx2::{compare_block3_f32_avx2, compare_block3_f64_avx2, compare_block4_f32_avx2};

#[cfg(not(any(target_feature = "avx2", target_feature = "avx512f")))]
compile_error!("DonnellyMarkerSimd on x86_64 requires AVX2 or AVX-512");
