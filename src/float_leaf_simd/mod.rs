pub(crate) mod fallback;
pub mod leaf_node;

// TODO: fix f32 AVX2

// #[cfg(all(
//     feature = "simd",
//     target_feature = "avx2",
//     any(target_arch = "x86", target_arch = "x86_64")
// ))]
// pub(crate) mod f32_avx2;

#[cfg(all(
    feature = "simd",
    target_feature = "avx2",
    any(target_arch = "x86", target_arch = "x86_64")
))]
pub(crate) mod f64_avx2;

// TODO: test f32 AVX512

// #[cfg(all(
//     feature = "simd",
//     target_feature = "avx512f",
//     any(target_arch = "x86", target_arch = "x86_64")
// ))]
// pub(crate) mod f32_avx512;

#[cfg(all(
    feature = "simd",
    target_feature = "avx512f",
    any(target_arch = "x86", target_arch = "x86_64")
))]
pub(crate) mod f64_avx512;
