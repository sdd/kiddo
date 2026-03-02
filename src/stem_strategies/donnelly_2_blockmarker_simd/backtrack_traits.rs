//! Type-specific SIMD backtrack mask generation traits for Donnelly block traversal.
//!
//! Replaces size_of-based dispatch with explicit trait implementations per type.
//! Each type (f32, f64, fixed-point, etc.) implements its own SIMD/autovec backtrack
//! strategy, and the metric is passed as a generic parameter for distance calculations.

use std::ptr::NonNull;

use crate::traits_unified_2::AxisUnified;
use crate::traits_unified_2::{DistanceMetricUnified, DotProduct, Manhattan, SquaredEuclidean};

mod sealed {
    pub trait Sealed {}
}

/// Metric-provided SIMD backtrack contract for Block3.
///
/// Implementors must provide an autovec/scalar baseline. Architecture-specific hooks
/// are optional and default to the baseline.
pub trait DistanceMetricSimdBlock3<A: Copy, const K: usize, O>
where
    O: AxisUnified<Coord = O>,
{
    /// Autovec/scalar baseline implementation for Block3 backtracking mask generation.
    fn backtrack_block3_autovec(
        query_wide: O,
        stems_ptr: NonNull<u8>,
        block_base_idx: usize,
        old_off: O,
        rd: O,
        best_dist: O,
    ) -> u8;

    #[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx2"))]
    #[inline(always)]
    /// AVX2 override for Block3. Defaults to autovec unless metric overrides it.
    unsafe fn backtrack_block3_avx2(
        query_wide: O,
        stems_ptr: NonNull<u8>,
        block_base_idx: usize,
        old_off: O,
        rd: O,
        best_dist: O,
    ) -> u8 {
        Self::backtrack_block3_autovec(
            query_wide,
            stems_ptr,
            block_base_idx,
            old_off,
            rd,
            best_dist,
        )
    }

    #[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx512f"))]
    #[inline(always)]
    /// AVX-512 override for Block3. Defaults to autovec unless metric overrides it.
    unsafe fn backtrack_block3_avx512(
        query_wide: O,
        stems_ptr: NonNull<u8>,
        block_base_idx: usize,
        old_off: O,
        rd: O,
        best_dist: O,
    ) -> u8 {
        Self::backtrack_block3_autovec(
            query_wide,
            stems_ptr,
            block_base_idx,
            old_off,
            rd,
            best_dist,
        )
    }

    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    #[inline(always)]
    /// NEON override for Block3. Defaults to autovec unless metric overrides it.
    unsafe fn backtrack_block3_neon(
        query_wide: O,
        stems_ptr: NonNull<u8>,
        block_base_idx: usize,
        old_off: O,
        rd: O,
        best_dist: O,
    ) -> u8 {
        Self::backtrack_block3_autovec(
            query_wide,
            stems_ptr,
            block_base_idx,
            old_off,
            rd,
            best_dist,
        )
    }

    #[inline(always)]
    /// Compile-time dispatch to the best available Block3 implementation.
    fn backtrack_block3(
        query_wide: O,
        stems_ptr: NonNull<u8>,
        block_base_idx: usize,
        old_off: O,
        rd: O,
        best_dist: O,
    ) -> u8 {
        #[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx512f"))]
        {
            // SAFETY: guarded by target cfg and delegated to metric hook.
            return unsafe {
                Self::backtrack_block3_avx512(
                    query_wide,
                    stems_ptr,
                    block_base_idx,
                    old_off,
                    rd,
                    best_dist,
                )
            };
        }

        #[cfg(all(
            feature = "simd",
            target_arch = "x86_64",
            target_feature = "avx2",
            not(target_feature = "avx512f")
        ))]
        {
            // SAFETY: guarded by target cfg and delegated to metric hook.
            return unsafe {
                Self::backtrack_block3_avx2(
                    query_wide,
                    stems_ptr,
                    block_base_idx,
                    old_off,
                    rd,
                    best_dist,
                )
            };
        }

        #[cfg(all(feature = "simd", target_arch = "aarch64"))]
        {
            // SAFETY: guarded by target cfg and delegated to metric hook.
            return unsafe {
                Self::backtrack_block3_neon(
                    query_wide,
                    stems_ptr,
                    block_base_idx,
                    old_off,
                    rd,
                    best_dist,
                )
            };
        }
        #[cfg(not(any(
            all(feature = "simd", target_arch = "x86_64", target_feature = "avx512f"),
            all(
                feature = "simd",
                target_arch = "x86_64",
                target_feature = "avx2",
                not(target_feature = "avx512f")
            ),
            all(feature = "simd", target_arch = "aarch64")
        )))]
        {
            Self::backtrack_block3_autovec(
                query_wide,
                stems_ptr,
                block_base_idx,
                old_off,
                rd,
                best_dist,
            )
        }
    }
}

/// Metric-provided SIMD backtrack contract for Block4.
pub trait DistanceMetricSimdBlock4<A: Copy, const K: usize, O>
where
    O: AxisUnified<Coord = O>,
{
    /// Autovec/scalar baseline implementation for Block4 backtracking mask generation.
    fn backtrack_block4_autovec(
        query_wide: O,
        stems_ptr: NonNull<u8>,
        block_base_idx: usize,
        old_off: O,
        rd: O,
        best_dist: O,
    ) -> u16;

    #[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx2"))]
    #[inline(always)]
    /// AVX2 override for Block4. Defaults to autovec unless metric overrides it.
    unsafe fn backtrack_block4_avx2(
        query_wide: O,
        stems_ptr: NonNull<u8>,
        block_base_idx: usize,
        old_off: O,
        rd: O,
        best_dist: O,
    ) -> u16 {
        Self::backtrack_block4_autovec(
            query_wide,
            stems_ptr,
            block_base_idx,
            old_off,
            rd,
            best_dist,
        )
    }

    #[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx512f"))]
    #[inline(always)]
    /// AVX-512 override for Block4. Defaults to autovec unless metric overrides it.
    unsafe fn backtrack_block4_avx512(
        query_wide: O,
        stems_ptr: NonNull<u8>,
        block_base_idx: usize,
        old_off: O,
        rd: O,
        best_dist: O,
    ) -> u16 {
        Self::backtrack_block4_autovec(
            query_wide,
            stems_ptr,
            block_base_idx,
            old_off,
            rd,
            best_dist,
        )
    }

    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    #[inline(always)]
    /// NEON override for Block4. Defaults to autovec unless metric overrides it.
    unsafe fn backtrack_block4_neon(
        query_wide: O,
        stems_ptr: NonNull<u8>,
        block_base_idx: usize,
        old_off: O,
        rd: O,
        best_dist: O,
    ) -> u16 {
        Self::backtrack_block4_autovec(
            query_wide,
            stems_ptr,
            block_base_idx,
            old_off,
            rd,
            best_dist,
        )
    }

    #[inline(always)]
    /// Compile-time dispatch to the best available Block4 implementation.
    fn backtrack_block4(
        query_wide: O,
        stems_ptr: NonNull<u8>,
        block_base_idx: usize,
        old_off: O,
        rd: O,
        best_dist: O,
    ) -> u16 {
        #[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx512f"))]
        {
            // SAFETY: guarded by target cfg and delegated to metric hook.
            return unsafe {
                Self::backtrack_block4_avx512(
                    query_wide,
                    stems_ptr,
                    block_base_idx,
                    old_off,
                    rd,
                    best_dist,
                )
            };
        }

        #[cfg(all(
            feature = "simd",
            target_arch = "x86_64",
            target_feature = "avx2",
            not(target_feature = "avx512f")
        ))]
        {
            // SAFETY: guarded by target cfg and delegated to metric hook.
            return unsafe {
                Self::backtrack_block4_avx2(
                    query_wide,
                    stems_ptr,
                    block_base_idx,
                    old_off,
                    rd,
                    best_dist,
                )
            };
        }

        #[cfg(all(feature = "simd", target_arch = "aarch64"))]
        {
            // SAFETY: guarded by target cfg and delegated to metric hook.
            return unsafe {
                Self::backtrack_block4_neon(
                    query_wide,
                    stems_ptr,
                    block_base_idx,
                    old_off,
                    rd,
                    best_dist,
                )
            };
        }
        #[cfg(not(any(
            all(feature = "simd", target_arch = "x86_64", target_feature = "avx512f"),
            all(
                feature = "simd",
                target_arch = "x86_64",
                target_feature = "avx2",
                not(target_feature = "avx512f")
            ),
            all(feature = "simd", target_arch = "aarch64")
        )))]
        {
            Self::backtrack_block4_autovec(
                query_wide,
                stems_ptr,
                block_base_idx,
                old_off,
                rd,
                best_dist,
            )
        }
    }
}

/// Trait for computing SIMD backtrack masks for Block3 (8 children).
///
/// The backtrack mask indicates which siblings need to be explored based on their
/// potential to contain closer points than the current best distance.
///
/// Default implementation panics - types must provide actual implementations.
pub trait BacktrackBlock3: AxisUnified<Coord = Self> + sealed::Sealed {
    /// Compute backtrack mask for Block3 using SIMD or autovec.
    ///
    /// # Arguments
    /// * `query_wide` - Query coordinate in this dimension (widened to output type)
    /// * `stems_ptr` - Pointer to start of stems array
    /// * `block_base_idx` - Cache-line base index for this block
    /// * `old_off` - Current offset contribution from this dimension
    /// * `rd` - Current running distance
    /// * `best_dist` - Best distance found so far
    ///
    /// # Returns
    /// Bitmask where bit i is set if child i needs exploration.
    ///
    /// # Type Parameters
    /// * `D` - Distance metric (e.g., SquaredEuclidean, Manhattan)
    /// * `K` - Number of dimensions (required by DistanceMetricUnified)
    fn backtrack_block3<A, D, const K: usize>(
        query_wide: Self,
        stems_ptr: NonNull<u8>,
        block_base_idx: usize,
        old_off: Self,
        rd: Self,
        best_dist: Self,
    ) -> u8
    where
        A: Copy,
        D: DistanceMetricUnified<A, K, Output = Self> + DistanceMetricSimdBlock3<A, K, Self>;
}

/// Trait for computing SIMD backtrack masks for Block4 (16 children).
///
/// Similar to BacktrackBlock3, but returns u16 mask for 16 children.
pub trait BacktrackBlock4: AxisUnified<Coord = Self> + sealed::Sealed {
    /// Compute backtrack mask for Block4 using SIMD or autovec.
    ///
    /// # Arguments
    /// Same as BacktrackBlock3.
    ///
    /// # Returns
    /// Bitmask (u16) where bit i is set if child i needs exploration.
    ///
    /// # Type Parameters
    /// * `D` - Distance metric (e.g., SquaredEuclidean, Manhattan)
    /// * `K` - Number of dimensions (required by DistanceMetricUnified)
    fn backtrack_block4<A, D, const K: usize>(
        query_wide: Self,
        stems_ptr: NonNull<u8>,
        block_base_idx: usize,
        old_off: Self,
        rd: Self,
        best_dist: Self,
    ) -> u16
    where
        A: Copy,
        D: DistanceMetricUnified<A, K, Output = Self> + DistanceMetricSimdBlock4<A, K, Self>;
}

// ====================================================================================
// Autovec fallback implementations
// ====================================================================================

/// Autovec fallback for Block3 backtrack mask computation.
///
/// Computes interval-based distance for each of 8 siblings and builds mask.
#[inline(always)]
fn autovec_backtrack_block3<A, I, D, const K: usize>(
    query_wide: A,
    stems_ptr: NonNull<u8>,
    block_base_idx: usize,
    old_off: A,
    rd: A,
    best_dist: A,
) -> u8
where
    A: AxisUnified<Coord = A> + std::ops::Add<Output = A> + std::ops::Sub<Output = A>,
    I: Copy,
    D: DistanceMetricUnified<I, K, Output = A>,
{
    use super::{child_interval_bounds_block3, interval_distance_1d};

    let mut mask: u8 = 0;

    for sibling_idx in 0..8u8 {
        let (lower_offset, upper_offset) = child_interval_bounds_block3(sibling_idx as usize);

        let ptr = unsafe {
            stems_ptr
                .as_ptr()
                .add(block_base_idx * std::mem::size_of::<A>()) as *const A
        };

        // Get pivot values for bounds (255 = ±∞)
        let lower_val = if lower_offset == 255 {
            A::min_value()
        } else {
            unsafe { *ptr.add(lower_offset as usize) }
        };

        let upper_val = if upper_offset == 255 {
            A::max_value()
        } else {
            unsafe { *ptr.add(upper_offset as usize) }
        };

        // Compute interval distance
        let interval_dist = interval_distance_1d::<A>(query_wide, lower_val, upper_val);

        // Update rd via per-dimension contribution swap:
        // rd_far = rd - dist1(old_off, 0) + dist1(new_off, 0)
        let old_dist1 = D::dist1(old_off, A::zero());
        let new_dist1 = D::dist1(interval_dist, A::zero());
        let rd_far = rd - old_dist1 + new_dist1;

        if rd_far <= best_dist {
            mask |= 1 << sibling_idx;
        }
    }

    mask
}

/// Autovec fallback for Block4 backtrack mask computation.
#[inline(always)]
fn autovec_backtrack_block4<A, I, D, const K: usize>(
    query_wide: A,
    stems_ptr: NonNull<u8>,
    block_base_idx: usize,
    old_off: A,
    rd: A,
    best_dist: A,
) -> u16
where
    A: AxisUnified<Coord = A> + std::ops::Add<Output = A> + std::ops::Sub<Output = A>,
    I: Copy,
    D: DistanceMetricUnified<I, K, Output = A>,
{
    use super::{child_interval_bounds_block4, interval_distance_1d};

    let mut mask: u16 = 0;

    for sibling_idx in 0..16u8 {
        let (lower_offset, upper_offset) = child_interval_bounds_block4(sibling_idx as usize);

        let ptr = unsafe {
            stems_ptr
                .as_ptr()
                .add(block_base_idx * std::mem::size_of::<A>()) as *const A
        };

        // Get pivot values for bounds (255 = ±∞)
        let lower_val = if lower_offset == 255 {
            A::min_value()
        } else {
            unsafe { *ptr.add(lower_offset as usize) }
        };

        let upper_val = if upper_offset == 255 {
            A::max_value()
        } else {
            unsafe { *ptr.add(upper_offset as usize) }
        };

        // Compute interval distance
        let interval_dist = interval_distance_1d::<A>(query_wide, lower_val, upper_val);

        // Update rd via per-dimension contribution swap:
        // rd_far = rd - dist1(old_off, 0) + dist1(new_off, 0)
        let old_dist1 = D::dist1(old_off, A::zero());
        let new_dist1 = D::dist1(interval_dist, A::zero());
        let rd_far = rd - old_dist1 + new_dist1;

        if rd_far <= best_dist {
            mask |= 1 << sibling_idx;
        }
    }

    mask
}

// ====================================================================================
// DistanceMetricSimdBlock* implementations for core float metrics
// ====================================================================================

impl<A, const K: usize> DistanceMetricSimdBlock3<A, K, f64> for SquaredEuclidean<f64>
where
    A: Copy,
    SquaredEuclidean<f64>: DistanceMetricUnified<A, K, Output = f64>,
{
    #[inline(always)]
    fn backtrack_block3_autovec(
        query_wide: f64,
        stems_ptr: NonNull<u8>,
        block_base_idx: usize,
        old_off: f64,
        rd: f64,
        best_dist: f64,
    ) -> u8 {
        autovec_backtrack_block3::<f64, A, Self, K>(
            query_wide,
            stems_ptr,
            block_base_idx,
            old_off,
            rd,
            best_dist,
        )
    }

    #[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx2"))]
    #[inline(always)]
    unsafe fn backtrack_block3_avx2(
        query_wide: f64,
        stems_ptr: NonNull<u8>,
        block_base_idx: usize,
        old_off: f64,
        rd: f64,
        best_dist: f64,
    ) -> u8 {
        simd_backtrack_block3_f64_avx2_squared_euclidean::<A, Self, K>(
            query_wide,
            stems_ptr,
            block_base_idx,
            old_off,
            rd,
            best_dist,
        )
    }

    #[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx512f"))]
    #[inline(always)]
    unsafe fn backtrack_block3_avx512(
        query_wide: f64,
        stems_ptr: NonNull<u8>,
        block_base_idx: usize,
        old_off: f64,
        rd: f64,
        best_dist: f64,
    ) -> u8 {
        simd_backtrack_block3_f64_avx512_squared_euclidean::<A, Self, K>(
            query_wide,
            stems_ptr,
            block_base_idx,
            old_off,
            rd,
            best_dist,
        )
    }

    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    #[inline(always)]
    unsafe fn backtrack_block3_neon(
        query_wide: f64,
        stems_ptr: NonNull<u8>,
        block_base_idx: usize,
        old_off: f64,
        rd: f64,
        best_dist: f64,
    ) -> u8 {
        simd_backtrack_block3_f64_neon_squared_euclidean::<A, Self, K>(
            query_wide,
            stems_ptr,
            block_base_idx,
            old_off,
            rd,
            best_dist,
        )
    }
}

impl<A, const K: usize> DistanceMetricSimdBlock4<A, K, f64> for SquaredEuclidean<f64>
where
    A: Copy,
    SquaredEuclidean<f64>: DistanceMetricUnified<A, K, Output = f64>,
{
    #[inline(always)]
    fn backtrack_block4_autovec(
        query_wide: f64,
        stems_ptr: NonNull<u8>,
        block_base_idx: usize,
        old_off: f64,
        rd: f64,
        best_dist: f64,
    ) -> u16 {
        autovec_backtrack_block4::<f64, A, Self, K>(
            query_wide,
            stems_ptr,
            block_base_idx,
            old_off,
            rd,
            best_dist,
        )
    }

    #[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx2"))]
    #[inline(always)]
    unsafe fn backtrack_block4_avx2(
        query_wide: f64,
        stems_ptr: NonNull<u8>,
        block_base_idx: usize,
        old_off: f64,
        rd: f64,
        best_dist: f64,
    ) -> u16 {
        simd_backtrack_block4_f64_avx2_squared_euclidean::<A, Self, K>(
            query_wide,
            stems_ptr,
            block_base_idx,
            old_off,
            rd,
            best_dist,
        )
    }

    #[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx512f"))]
    #[inline(always)]
    unsafe fn backtrack_block4_avx512(
        query_wide: f64,
        stems_ptr: NonNull<u8>,
        block_base_idx: usize,
        old_off: f64,
        rd: f64,
        best_dist: f64,
    ) -> u16 {
        simd_backtrack_block4_f64_avx512_squared_euclidean::<A, Self, K>(
            query_wide,
            stems_ptr,
            block_base_idx,
            old_off,
            rd,
            best_dist,
        )
    }

    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    #[inline(always)]
    unsafe fn backtrack_block4_neon(
        query_wide: f64,
        stems_ptr: NonNull<u8>,
        block_base_idx: usize,
        old_off: f64,
        rd: f64,
        best_dist: f64,
    ) -> u16 {
        simd_backtrack_block4_f64_neon_squared_euclidean::<A, Self, K>(
            query_wide,
            stems_ptr,
            block_base_idx,
            old_off,
            rd,
            best_dist,
        )
    }
}

impl<A, const K: usize> DistanceMetricSimdBlock3<A, K, f32> for SquaredEuclidean<f32>
where
    A: Copy,
    SquaredEuclidean<f32>: DistanceMetricUnified<A, K, Output = f32>,
{
    #[inline(always)]
    fn backtrack_block3_autovec(
        query_wide: f32,
        stems_ptr: NonNull<u8>,
        block_base_idx: usize,
        old_off: f32,
        rd: f32,
        best_dist: f32,
    ) -> u8 {
        autovec_backtrack_block3::<f32, A, Self, K>(
            query_wide,
            stems_ptr,
            block_base_idx,
            old_off,
            rd,
            best_dist,
        )
    }

    #[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx2"))]
    #[inline(always)]
    unsafe fn backtrack_block3_avx2(
        query_wide: f32,
        stems_ptr: NonNull<u8>,
        block_base_idx: usize,
        old_off: f32,
        rd: f32,
        best_dist: f32,
    ) -> u8 {
        simd_backtrack_block3_f32_avx2_squared_euclidean::<A, Self, K>(
            query_wide,
            stems_ptr,
            block_base_idx,
            old_off,
            rd,
            best_dist,
        )
    }

    #[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx512f"))]
    #[inline(always)]
    unsafe fn backtrack_block3_avx512(
        query_wide: f32,
        stems_ptr: NonNull<u8>,
        block_base_idx: usize,
        old_off: f32,
        rd: f32,
        best_dist: f32,
    ) -> u8 {
        simd_backtrack_block3_f32_avx512_squared_euclidean::<A, Self, K>(
            query_wide,
            stems_ptr,
            block_base_idx,
            old_off,
            rd,
            best_dist,
        )
    }

    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    #[inline(always)]
    unsafe fn backtrack_block3_neon(
        query_wide: f32,
        stems_ptr: NonNull<u8>,
        block_base_idx: usize,
        old_off: f32,
        rd: f32,
        best_dist: f32,
    ) -> u8 {
        simd_backtrack_block3_f32_neon_squared_euclidean::<A, Self, K>(
            query_wide,
            stems_ptr,
            block_base_idx,
            old_off,
            rd,
            best_dist,
        )
    }
}

impl<A, const K: usize> DistanceMetricSimdBlock4<A, K, f32> for SquaredEuclidean<f32>
where
    A: Copy,
    SquaredEuclidean<f32>: DistanceMetricUnified<A, K, Output = f32>,
{
    #[inline(always)]
    fn backtrack_block4_autovec(
        query_wide: f32,
        stems_ptr: NonNull<u8>,
        block_base_idx: usize,
        old_off: f32,
        rd: f32,
        best_dist: f32,
    ) -> u16 {
        autovec_backtrack_block4::<f32, A, Self, K>(
            query_wide,
            stems_ptr,
            block_base_idx,
            old_off,
            rd,
            best_dist,
        )
    }

    #[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx2"))]
    #[inline(always)]
    unsafe fn backtrack_block4_avx2(
        query_wide: f32,
        stems_ptr: NonNull<u8>,
        block_base_idx: usize,
        old_off: f32,
        rd: f32,
        best_dist: f32,
    ) -> u16 {
        simd_backtrack_block4_f32_avx2_squared_euclidean::<A, Self, K>(
            query_wide,
            stems_ptr,
            block_base_idx,
            old_off,
            rd,
            best_dist,
        )
    }

    #[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx512f"))]
    #[inline(always)]
    unsafe fn backtrack_block4_avx512(
        query_wide: f32,
        stems_ptr: NonNull<u8>,
        block_base_idx: usize,
        old_off: f32,
        rd: f32,
        best_dist: f32,
    ) -> u16 {
        simd_backtrack_block4_f32_avx512_squared_euclidean::<A, Self, K>(
            query_wide,
            stems_ptr,
            block_base_idx,
            old_off,
            rd,
            best_dist,
        )
    }

    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    #[inline(always)]
    unsafe fn backtrack_block4_neon(
        query_wide: f32,
        stems_ptr: NonNull<u8>,
        block_base_idx: usize,
        old_off: f32,
        rd: f32,
        best_dist: f32,
    ) -> u16 {
        simd_backtrack_block4_f32_neon_squared_euclidean::<A, Self, K>(
            query_wide,
            stems_ptr,
            block_base_idx,
            old_off,
            rd,
            best_dist,
        )
    }
}

impl<A, const K: usize> DistanceMetricSimdBlock3<A, K, f64> for Manhattan<f64>
where
    A: Copy,
    Manhattan<f64>: DistanceMetricUnified<A, K, Output = f64>,
{
    #[inline(always)]
    fn backtrack_block3_autovec(
        query_wide: f64,
        stems_ptr: NonNull<u8>,
        block_base_idx: usize,
        old_off: f64,
        rd: f64,
        best_dist: f64,
    ) -> u8 {
        autovec_backtrack_block3::<f64, A, Self, K>(
            query_wide,
            stems_ptr,
            block_base_idx,
            old_off,
            rd,
            best_dist,
        )
    }

    #[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx2"))]
    #[inline(always)]
    unsafe fn backtrack_block3_avx2(
        query_wide: f64,
        stems_ptr: NonNull<u8>,
        block_base_idx: usize,
        old_off: f64,
        rd: f64,
        best_dist: f64,
    ) -> u8 {
        simd_backtrack_block3_f64_avx2_manhattan::<A, Self, K>(
            query_wide,
            stems_ptr,
            block_base_idx,
            old_off,
            rd,
            best_dist,
        )
    }

    #[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx512f"))]
    #[inline(always)]
    unsafe fn backtrack_block3_avx512(
        query_wide: f64,
        stems_ptr: NonNull<u8>,
        block_base_idx: usize,
        old_off: f64,
        rd: f64,
        best_dist: f64,
    ) -> u8 {
        simd_backtrack_block3_f64_avx512_manhattan::<A, Self, K>(
            query_wide,
            stems_ptr,
            block_base_idx,
            old_off,
            rd,
            best_dist,
        )
    }

    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    #[inline(always)]
    unsafe fn backtrack_block3_neon(
        query_wide: f64,
        stems_ptr: NonNull<u8>,
        block_base_idx: usize,
        old_off: f64,
        rd: f64,
        best_dist: f64,
    ) -> u8 {
        simd_backtrack_block3_f64_neon_manhattan::<A, Self, K>(
            query_wide,
            stems_ptr,
            block_base_idx,
            old_off,
            rd,
            best_dist,
        )
    }
}

impl<A, const K: usize> DistanceMetricSimdBlock4<A, K, f64> for Manhattan<f64>
where
    A: Copy,
    Manhattan<f64>: DistanceMetricUnified<A, K, Output = f64>,
{
    #[inline(always)]
    fn backtrack_block4_autovec(
        query_wide: f64,
        stems_ptr: NonNull<u8>,
        block_base_idx: usize,
        old_off: f64,
        rd: f64,
        best_dist: f64,
    ) -> u16 {
        autovec_backtrack_block4::<f64, A, Self, K>(
            query_wide,
            stems_ptr,
            block_base_idx,
            old_off,
            rd,
            best_dist,
        )
    }

    #[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx2"))]
    #[inline(always)]
    unsafe fn backtrack_block4_avx2(
        query_wide: f64,
        stems_ptr: NonNull<u8>,
        block_base_idx: usize,
        old_off: f64,
        rd: f64,
        best_dist: f64,
    ) -> u16 {
        simd_backtrack_block4_f64_avx2_manhattan::<A, Self, K>(
            query_wide,
            stems_ptr,
            block_base_idx,
            old_off,
            rd,
            best_dist,
        )
    }

    #[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx512f"))]
    #[inline(always)]
    unsafe fn backtrack_block4_avx512(
        query_wide: f64,
        stems_ptr: NonNull<u8>,
        block_base_idx: usize,
        old_off: f64,
        rd: f64,
        best_dist: f64,
    ) -> u16 {
        simd_backtrack_block4_f64_avx512_manhattan::<A, Self, K>(
            query_wide,
            stems_ptr,
            block_base_idx,
            old_off,
            rd,
            best_dist,
        )
    }

    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    #[inline(always)]
    unsafe fn backtrack_block4_neon(
        query_wide: f64,
        stems_ptr: NonNull<u8>,
        block_base_idx: usize,
        old_off: f64,
        rd: f64,
        best_dist: f64,
    ) -> u16 {
        simd_backtrack_block4_f64_neon_manhattan::<A, Self, K>(
            query_wide,
            stems_ptr,
            block_base_idx,
            old_off,
            rd,
            best_dist,
        )
    }
}

impl<A, const K: usize> DistanceMetricSimdBlock3<A, K, f32> for Manhattan<f32>
where
    A: Copy,
    Manhattan<f32>: DistanceMetricUnified<A, K, Output = f32>,
{
    #[inline(always)]
    fn backtrack_block3_autovec(
        query_wide: f32,
        stems_ptr: NonNull<u8>,
        block_base_idx: usize,
        old_off: f32,
        rd: f32,
        best_dist: f32,
    ) -> u8 {
        autovec_backtrack_block3::<f32, A, Self, K>(
            query_wide,
            stems_ptr,
            block_base_idx,
            old_off,
            rd,
            best_dist,
        )
    }

    #[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx2"))]
    #[inline(always)]
    unsafe fn backtrack_block3_avx2(
        query_wide: f32,
        stems_ptr: NonNull<u8>,
        block_base_idx: usize,
        old_off: f32,
        rd: f32,
        best_dist: f32,
    ) -> u8 {
        simd_backtrack_block3_f32_avx2_manhattan::<A, Self, K>(
            query_wide,
            stems_ptr,
            block_base_idx,
            old_off,
            rd,
            best_dist,
        )
    }

    #[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx512f"))]
    #[inline(always)]
    unsafe fn backtrack_block3_avx512(
        query_wide: f32,
        stems_ptr: NonNull<u8>,
        block_base_idx: usize,
        old_off: f32,
        rd: f32,
        best_dist: f32,
    ) -> u8 {
        simd_backtrack_block3_f32_avx512_manhattan::<A, Self, K>(
            query_wide,
            stems_ptr,
            block_base_idx,
            old_off,
            rd,
            best_dist,
        )
    }

    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    #[inline(always)]
    unsafe fn backtrack_block3_neon(
        query_wide: f32,
        stems_ptr: NonNull<u8>,
        block_base_idx: usize,
        old_off: f32,
        rd: f32,
        best_dist: f32,
    ) -> u8 {
        simd_backtrack_block3_f32_neon_manhattan::<A, Self, K>(
            query_wide,
            stems_ptr,
            block_base_idx,
            old_off,
            rd,
            best_dist,
        )
    }
}

impl<A, const K: usize> DistanceMetricSimdBlock4<A, K, f32> for Manhattan<f32>
where
    A: Copy,
    Manhattan<f32>: DistanceMetricUnified<A, K, Output = f32>,
{
    #[inline(always)]
    fn backtrack_block4_autovec(
        query_wide: f32,
        stems_ptr: NonNull<u8>,
        block_base_idx: usize,
        old_off: f32,
        rd: f32,
        best_dist: f32,
    ) -> u16 {
        autovec_backtrack_block4::<f32, A, Self, K>(
            query_wide,
            stems_ptr,
            block_base_idx,
            old_off,
            rd,
            best_dist,
        )
    }

    #[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx2"))]
    #[inline(always)]
    unsafe fn backtrack_block4_avx2(
        query_wide: f32,
        stems_ptr: NonNull<u8>,
        block_base_idx: usize,
        old_off: f32,
        rd: f32,
        best_dist: f32,
    ) -> u16 {
        simd_backtrack_block4_f32_avx2_manhattan::<A, Self, K>(
            query_wide,
            stems_ptr,
            block_base_idx,
            old_off,
            rd,
            best_dist,
        )
    }

    #[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx512f"))]
    #[inline(always)]
    unsafe fn backtrack_block4_avx512(
        query_wide: f32,
        stems_ptr: NonNull<u8>,
        block_base_idx: usize,
        old_off: f32,
        rd: f32,
        best_dist: f32,
    ) -> u16 {
        simd_backtrack_block4_f32_avx512_manhattan::<A, Self, K>(
            query_wide,
            stems_ptr,
            block_base_idx,
            old_off,
            rd,
            best_dist,
        )
    }

    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    #[inline(always)]
    unsafe fn backtrack_block4_neon(
        query_wide: f32,
        stems_ptr: NonNull<u8>,
        block_base_idx: usize,
        old_off: f32,
        rd: f32,
        best_dist: f32,
    ) -> u16 {
        simd_backtrack_block4_f32_neon_manhattan::<A, Self, K>(
            query_wide,
            stems_ptr,
            block_base_idx,
            old_off,
            rd,
            best_dist,
        )
    }
}

// DotProduct support remains autovec-only for now.
impl<A, const K: usize> DistanceMetricSimdBlock3<A, K, f64> for DotProduct<f64>
where
    A: Copy,
    DotProduct<f64>: DistanceMetricUnified<A, K, Output = f64>,
{
    #[inline(always)]
    fn backtrack_block3_autovec(
        _query_wide: f64,
        _stems_ptr: NonNull<u8>,
        _block_base_idx: usize,
        _old_off: f64,
        _rd: f64,
        _best_dist: f64,
    ) -> u8 {
        panic!("DotProduct is not currently supported with Donnelly SIMD backtracking");
    }
}

impl<A, const K: usize> DistanceMetricSimdBlock4<A, K, f64> for DotProduct<f64>
where
    A: Copy,
    DotProduct<f64>: DistanceMetricUnified<A, K, Output = f64>,
{
    #[inline(always)]
    fn backtrack_block4_autovec(
        _query_wide: f64,
        _stems_ptr: NonNull<u8>,
        _block_base_idx: usize,
        _old_off: f64,
        _rd: f64,
        _best_dist: f64,
    ) -> u16 {
        panic!("DotProduct is not currently supported with Donnelly SIMD backtracking");
    }
}

impl<A, const K: usize> DistanceMetricSimdBlock3<A, K, f32> for DotProduct<f32>
where
    A: Copy,
    DotProduct<f32>: DistanceMetricUnified<A, K, Output = f32>,
{
    #[inline(always)]
    fn backtrack_block3_autovec(
        _query_wide: f32,
        _stems_ptr: NonNull<u8>,
        _block_base_idx: usize,
        _old_off: f32,
        _rd: f32,
        _best_dist: f32,
    ) -> u8 {
        panic!("DotProduct is not currently supported with Donnelly SIMD backtracking");
    }
}

impl<A, const K: usize> DistanceMetricSimdBlock4<A, K, f32> for DotProduct<f32>
where
    A: Copy,
    DotProduct<f32>: DistanceMetricUnified<A, K, Output = f32>,
{
    #[inline(always)]
    fn backtrack_block4_autovec(
        _query_wide: f32,
        _stems_ptr: NonNull<u8>,
        _block_base_idx: usize,
        _old_off: f32,
        _rd: f32,
        _best_dist: f32,
    ) -> u16 {
        panic!("DotProduct is not currently supported with Donnelly SIMD backtracking");
    }
}

// ====================================================================================
// f64 implementations
// ====================================================================================

impl sealed::Sealed for f64 {}

impl BacktrackBlock3 for f64 {
    #[inline(always)]
    fn backtrack_block3<A, D, const K: usize>(
        query_wide: Self,
        stems_ptr: NonNull<u8>,
        block_base_idx: usize,
        old_off: Self,
        rd: Self,
        best_dist: Self,
    ) -> u8
    where
        A: Copy,
        D: DistanceMetricUnified<A, K, Output = Self> + DistanceMetricSimdBlock3<A, K, Self>,
    {
        D::backtrack_block3(
            query_wide,
            stems_ptr,
            block_base_idx,
            old_off,
            rd,
            best_dist,
        )
    }
}

impl BacktrackBlock4 for f64 {
    #[inline(always)]
    fn backtrack_block4<A, D, const K: usize>(
        query_wide: Self,
        stems_ptr: NonNull<u8>,
        block_base_idx: usize,
        old_off: Self,
        rd: Self,
        best_dist: Self,
    ) -> u16
    where
        A: Copy,
        D: DistanceMetricUnified<A, K, Output = Self> + DistanceMetricSimdBlock4<A, K, Self>,
    {
        D::backtrack_block4(
            query_wide,
            stems_ptr,
            block_base_idx,
            old_off,
            rd,
            best_dist,
        )
    }
}

// ====================================================================================
// f32 implementations
// ====================================================================================

impl sealed::Sealed for f32 {}

impl BacktrackBlock3 for f32 {
    #[inline(always)]
    fn backtrack_block3<A, D, const K: usize>(
        query_wide: Self,
        stems_ptr: NonNull<u8>,
        block_base_idx: usize,
        old_off: Self,
        rd: Self,
        best_dist: Self,
    ) -> u8
    where
        A: Copy,
        D: DistanceMetricUnified<A, K, Output = Self> + DistanceMetricSimdBlock3<A, K, Self>,
    {
        D::backtrack_block3(
            query_wide,
            stems_ptr,
            block_base_idx,
            old_off,
            rd,
            best_dist,
        )
    }
}

impl BacktrackBlock4 for f32 {
    #[inline(always)]
    fn backtrack_block4<A, D, const K: usize>(
        query_wide: Self,
        stems_ptr: NonNull<u8>,
        block_base_idx: usize,
        old_off: Self,
        rd: Self,
        best_dist: Self,
    ) -> u16
    where
        A: Copy,
        D: DistanceMetricUnified<A, K, Output = Self> + DistanceMetricSimdBlock4<A, K, Self>,
    {
        D::backtrack_block4(
            query_wide,
            stems_ptr,
            block_base_idx,
            old_off,
            rd,
            best_dist,
        )
    }
}

// ====================================================================================
// x86_64 AVX2 implementations
// ====================================================================================

#[cfg(all(
    feature = "simd",
    target_arch = "x86_64",
    not(target_feature = "avx512f")
))]
#[inline(always)]
unsafe fn simd_backtrack_block3_f64_avx2_squared_euclidean<
    A: Copy,
    D: DistanceMetricUnified<A, K, Output = f64>,
    const K: usize,
>(
    query_wide: f64,
    stems_ptr: NonNull<u8>,
    block_base_idx: usize,
    old_off: f64,
    rd: f64,
    best_dist: f64,
) -> u8 {
    use std::arch::x86_64::*;
    let _ = core::marker::PhantomData::<D>;

    let ptr = stems_ptr.as_ptr().add(block_base_idx * 8) as *const f64;

    // Load pivots contiguously, then reshuffle into child-order interval bounds.
    let pivots_lo = _mm256_loadu_pd(ptr);
    let pivots_hi = _mm256_loadu_pd(ptr.add(4));
    let neg_inf = _mm256_set1_pd(f64::NEG_INFINITY);
    let pos_inf = _mm256_set1_pd(f64::INFINITY);
    let p4_broadcast = _mm256_permute4x64_pd(pivots_hi, 0x00);

    let lower_lo_base = _mm256_permute4x64_pd(pivots_lo, 0x1C);
    let lower_lo_inf_p4 = _mm256_blend_pd(neg_inf, p4_broadcast, 0b1000);
    let lower_lo = _mm256_blend_pd(lower_lo_base, lower_lo_inf_p4, 0b1001);

    let lower_hi_mix = _mm256_permute4x64_pd(pivots_hi, 0x84);
    let lower_hi = _mm256_blend_pd(pivots_lo, lower_hi_mix, 0b1010);

    let upper_lo_base = _mm256_permute4x64_pd(pivots_lo, 0x27);
    let upper_lo = _mm256_blend_pd(upper_lo_base, p4_broadcast, 0b0100);

    let upper_hi_lo = _mm256_permute4x64_pd(pivots_lo, 0x08);
    let upper_hi_hi = _mm256_permute4x64_pd(pivots_hi, 0x21);
    let upper_hi_base = _mm256_blend_pd(upper_hi_lo, upper_hi_hi, 0b0101);
    let upper_hi = _mm256_blend_pd(upper_hi_base, pos_inf, 0b1000);

    let query_vec = _mm256_set1_pd(query_wide);
    let old_off_sq_vec = _mm256_set1_pd(old_off * old_off);
    let rd_vec = _mm256_set1_pd(rd);
    let best_dist_vec = _mm256_set1_pd(best_dist);
    let zero_vec = _mm256_setzero_pd();

    let below_lo = _mm256_max_pd(_mm256_sub_pd(lower_lo, query_vec), zero_vec);
    let above_lo = _mm256_max_pd(_mm256_sub_pd(query_vec, upper_lo), zero_vec);
    let interval_lo = _mm256_add_pd(below_lo, above_lo);
    let new_sq_lo = _mm256_mul_pd(interval_lo, interval_lo);
    let rd_far_lo = _mm256_add_pd(rd_vec, _mm256_sub_pd(new_sq_lo, old_off_sq_vec));
    let mask_lo = _mm256_movemask_pd(_mm256_cmp_pd(rd_far_lo, best_dist_vec, _CMP_LE_OQ)) as u8;

    let below_hi = _mm256_max_pd(_mm256_sub_pd(lower_hi, query_vec), zero_vec);
    let above_hi = _mm256_max_pd(_mm256_sub_pd(query_vec, upper_hi), zero_vec);
    let interval_hi = _mm256_add_pd(below_hi, above_hi);
    let new_sq_hi = _mm256_mul_pd(interval_hi, interval_hi);
    let rd_far_hi = _mm256_add_pd(rd_vec, _mm256_sub_pd(new_sq_hi, old_off_sq_vec));
    let mask_hi = _mm256_movemask_pd(_mm256_cmp_pd(rd_far_hi, best_dist_vec, _CMP_LE_OQ)) as u8;

    mask_lo | (mask_hi << 4)
}

#[cfg(all(
    feature = "simd",
    target_arch = "x86_64",
    not(target_feature = "avx512f")
))]
#[inline(always)]
unsafe fn simd_backtrack_block3_f32_avx2_squared_euclidean<
    A: Copy,
    D: DistanceMetricUnified<A, K, Output = f32>,
    const K: usize,
>(
    query_wide: f32,
    stems_ptr: NonNull<u8>,
    block_base_idx: usize,
    old_off: f32,
    rd: f32,
    best_dist: f32,
) -> u8 {
    use std::arch::x86_64::*;
    let _ = core::marker::PhantomData::<D>;

    // Load 8 pivots into a scalar array
    let ptr = stems_ptr.as_ptr().add(block_base_idx * 4) as *const f32;
    let mut pivots = [0.0f32; 8];
    std::ptr::copy_nonoverlapping(ptr, pivots.as_mut_ptr(), 8);

    // Precompute child-indexed lower/upper bounds
    let mut lower_vals = [0.0f32; 8];
    let mut upper_vals = [0.0f32; 8];
    for i in 0..8 {
        let (lower_offset, upper_offset) = super::child_interval_bounds_block3(i);
        lower_vals[i] = if lower_offset == 255 {
            f32::NEG_INFINITY
        } else {
            pivots[lower_offset as usize]
        };
        upper_vals[i] = if upper_offset == 255 {
            f32::INFINITY
        } else {
            pivots[upper_offset as usize]
        };
    }

    // Broadcast values
    let lower = _mm256_loadu_ps(lower_vals.as_ptr());
    let upper = _mm256_loadu_ps(upper_vals.as_ptr());
    let query_vec = _mm256_set1_ps(query_wide);
    let old_off_sq_vec = _mm256_set1_ps(old_off * old_off);
    let rd_vec = _mm256_set1_ps(rd);
    let best_dist_vec = _mm256_set1_ps(best_dist);
    let zero_vec = _mm256_setzero_ps();

    let below = _mm256_max_ps(_mm256_sub_ps(lower, query_vec), zero_vec);
    let above = _mm256_max_ps(_mm256_sub_ps(query_vec, upper), zero_vec);
    let interval = _mm256_add_ps(below, above);
    let new_sq = _mm256_mul_ps(interval, interval);
    let rd_far = _mm256_add_ps(rd_vec, _mm256_sub_ps(new_sq, old_off_sq_vec));

    _mm256_movemask_ps(_mm256_cmp_ps(rd_far, best_dist_vec, _CMP_LE_OQ)) as u8
}

#[cfg(all(
    feature = "simd",
    target_arch = "x86_64",
    not(target_feature = "avx512f")
))]
#[inline(always)]
unsafe fn simd_backtrack_block4_f64_avx2_squared_euclidean<
    A: Copy,
    D: DistanceMetricUnified<A, K, Output = f64>,
    const K: usize,
>(
    query_wide: f64,
    stems_ptr: NonNull<u8>,
    block_base_idx: usize,
    old_off: f64,
    rd: f64,
    best_dist: f64,
) -> u16 {
    use std::arch::x86_64::*;
    let _ = core::marker::PhantomData::<D>;

    // Load 16 pivots into a scalar array
    let ptr = stems_ptr.as_ptr().add(block_base_idx * 8) as *const f64;
    let mut pivots = [0.0f64; 16];
    std::ptr::copy_nonoverlapping(ptr, pivots.as_mut_ptr(), 16);

    // Precompute child-indexed lower/upper bounds
    let mut lower_vals = [0.0f64; 16];
    let mut upper_vals = [0.0f64; 16];
    for i in 0..16 {
        let (lower_offset, upper_offset) = super::child_interval_bounds_block4(i);
        lower_vals[i] = if lower_offset == 255 {
            f64::NEG_INFINITY
        } else {
            pivots[lower_offset as usize]
        };
        upper_vals[i] = if upper_offset == 255 {
            f64::INFINITY
        } else {
            pivots[upper_offset as usize]
        };
    }

    let query_vec = _mm256_set1_pd(query_wide);
    let old_off_sq_vec = _mm256_set1_pd(old_off * old_off);
    let rd_vec = _mm256_set1_pd(rd);
    let best_dist_vec = _mm256_set1_pd(best_dist);
    let zero_vec = _mm256_setzero_pd();

    let mut mask: u16 = 0;
    for chunk in 0..4 {
        let idx = chunk * 4;
        let lower = _mm256_loadu_pd(lower_vals.as_ptr().add(idx));
        let upper = _mm256_loadu_pd(upper_vals.as_ptr().add(idx));

        let below = _mm256_max_pd(_mm256_sub_pd(lower, query_vec), zero_vec);
        let above = _mm256_max_pd(_mm256_sub_pd(query_vec, upper), zero_vec);
        let interval = _mm256_add_pd(below, above);

        let new_sq = _mm256_mul_pd(interval, interval);
        let rd_far = _mm256_add_pd(rd_vec, _mm256_sub_pd(new_sq, old_off_sq_vec));

        let chunk_mask = _mm256_movemask_pd(_mm256_cmp_pd(rd_far, best_dist_vec, _CMP_LE_OQ));
        mask |= (chunk_mask as u16) << (chunk * 4);
    }

    mask
}

#[cfg(all(
    feature = "simd",
    target_arch = "x86_64",
    not(target_feature = "avx512f")
))]
#[inline(always)]
unsafe fn simd_backtrack_block4_f32_avx2_squared_euclidean<
    A: Copy,
    D: DistanceMetricUnified<A, K, Output = f32>,
    const K: usize,
>(
    query_wide: f32,
    stems_ptr: NonNull<u8>,
    block_base_idx: usize,
    old_off: f32,
    rd: f32,
    best_dist: f32,
) -> u16 {
    use std::arch::x86_64::*;
    let _ = core::marker::PhantomData::<D>;

    // Load 16 pivots into a scalar array
    let ptr = stems_ptr.as_ptr().add(block_base_idx * 4) as *const f32;
    let mut pivots = [0.0f32; 16];
    std::ptr::copy_nonoverlapping(ptr, pivots.as_mut_ptr(), 16);

    // Precompute child-indexed lower/upper bounds
    let mut lower_vals = [0.0f32; 16];
    let mut upper_vals = [0.0f32; 16];
    for i in 0..16 {
        let (lower_offset, upper_offset) = super::child_interval_bounds_block4(i);
        lower_vals[i] = if lower_offset == 255 {
            f32::NEG_INFINITY
        } else {
            pivots[lower_offset as usize]
        };
        upper_vals[i] = if upper_offset == 255 {
            f32::INFINITY
        } else {
            pivots[upper_offset as usize]
        };
    }

    let query_vec = _mm256_set1_ps(query_wide);
    let old_off_sq_vec = _mm256_set1_ps(old_off * old_off);
    let rd_vec = _mm256_set1_ps(rd);
    let best_dist_vec = _mm256_set1_ps(best_dist);
    let zero_vec = _mm256_setzero_ps();

    let lower_lo = _mm256_loadu_ps(lower_vals.as_ptr());
    let upper_lo = _mm256_loadu_ps(upper_vals.as_ptr());
    let below_lo = _mm256_max_ps(_mm256_sub_ps(lower_lo, query_vec), zero_vec);
    let above_lo = _mm256_max_ps(_mm256_sub_ps(query_vec, upper_lo), zero_vec);
    let interval_lo = _mm256_add_ps(below_lo, above_lo);
    let new_sq_lo = _mm256_mul_ps(interval_lo, interval_lo);
    let rd_far_lo = _mm256_add_ps(rd_vec, _mm256_sub_ps(new_sq_lo, old_off_sq_vec));
    let mask_lo = _mm256_movemask_ps(_mm256_cmp_ps(rd_far_lo, best_dist_vec, _CMP_LE_OQ)) as u16;

    let lower_hi = _mm256_loadu_ps(lower_vals.as_ptr().add(8));
    let upper_hi = _mm256_loadu_ps(upper_vals.as_ptr().add(8));
    let below_hi = _mm256_max_ps(_mm256_sub_ps(lower_hi, query_vec), zero_vec);
    let above_hi = _mm256_max_ps(_mm256_sub_ps(query_vec, upper_hi), zero_vec);
    let interval_hi = _mm256_add_ps(below_hi, above_hi);
    let new_sq_hi = _mm256_mul_ps(interval_hi, interval_hi);
    let rd_far_hi = _mm256_add_ps(rd_vec, _mm256_sub_ps(new_sq_hi, old_off_sq_vec));
    let mask_hi = _mm256_movemask_ps(_mm256_cmp_ps(rd_far_hi, best_dist_vec, _CMP_LE_OQ)) as u16;

    mask_lo | (mask_hi << 8)
}

#[cfg(all(
    feature = "simd",
    target_arch = "x86_64",
    not(target_feature = "avx512f")
))]
#[inline(always)]
unsafe fn simd_backtrack_block3_f64_avx2_manhattan<
    A: Copy,
    D: DistanceMetricUnified<A, K, Output = f64>,
    const K: usize,
>(
    query_wide: f64,
    stems_ptr: NonNull<u8>,
    block_base_idx: usize,
    old_off: f64,
    rd: f64,
    best_dist: f64,
) -> u8 {
    use std::arch::x86_64::*;
    let _ = core::marker::PhantomData::<D>;

    let pivots_ptr = stems_ptr.as_ptr().add(block_base_idx * 8) as *const f64;
    let mut pivots = [0.0f64; 8];
    std::ptr::copy_nonoverlapping(pivots_ptr, pivots.as_mut_ptr(), 8);

    let mut lower_vals = [0.0f64; 8];
    let mut upper_vals = [0.0f64; 8];
    for i in 0..8 {
        let (lower_offset, upper_offset) = super::child_interval_bounds_block3(i);
        lower_vals[i] = if lower_offset == 255 {
            f64::NEG_INFINITY
        } else {
            pivots[lower_offset as usize]
        };
        upper_vals[i] = if upper_offset == 255 {
            f64::INFINITY
        } else {
            pivots[upper_offset as usize]
        };
    }

    let query_vec = _mm256_set1_pd(query_wide);
    let old_off_abs_vec = _mm256_set1_pd(old_off.abs());
    let rd_vec = _mm256_set1_pd(rd);
    let best_dist_vec = _mm256_set1_pd(best_dist);
    let zero_vec = _mm256_setzero_pd();

    let lower_lo = _mm256_loadu_pd(lower_vals.as_ptr());
    let upper_lo = _mm256_loadu_pd(upper_vals.as_ptr());
    let below_lo = _mm256_max_pd(_mm256_sub_pd(lower_lo, query_vec), zero_vec);
    let above_lo = _mm256_max_pd(_mm256_sub_pd(query_vec, upper_lo), zero_vec);
    let interval_lo = _mm256_add_pd(below_lo, above_lo);
    let rd_far_lo = _mm256_add_pd(rd_vec, _mm256_sub_pd(interval_lo, old_off_abs_vec));
    let mask_lo = _mm256_movemask_pd(_mm256_cmp_pd(rd_far_lo, best_dist_vec, _CMP_LE_OQ)) as u8;

    let lower_hi = _mm256_loadu_pd(lower_vals.as_ptr().add(4));
    let upper_hi = _mm256_loadu_pd(upper_vals.as_ptr().add(4));
    let below_hi = _mm256_max_pd(_mm256_sub_pd(lower_hi, query_vec), zero_vec);
    let above_hi = _mm256_max_pd(_mm256_sub_pd(query_vec, upper_hi), zero_vec);
    let interval_hi = _mm256_add_pd(below_hi, above_hi);
    let rd_far_hi = _mm256_add_pd(rd_vec, _mm256_sub_pd(interval_hi, old_off_abs_vec));
    let mask_hi = _mm256_movemask_pd(_mm256_cmp_pd(rd_far_hi, best_dist_vec, _CMP_LE_OQ)) as u8;

    mask_lo | (mask_hi << 4)
}

#[cfg(all(
    feature = "simd",
    target_arch = "x86_64",
    not(target_feature = "avx512f")
))]
#[inline(always)]
unsafe fn simd_backtrack_block3_f32_avx2_manhattan<
    A: Copy,
    D: DistanceMetricUnified<A, K, Output = f32>,
    const K: usize,
>(
    query_wide: f32,
    stems_ptr: NonNull<u8>,
    block_base_idx: usize,
    old_off: f32,
    rd: f32,
    best_dist: f32,
) -> u8 {
    use std::arch::x86_64::*;
    let _ = core::marker::PhantomData::<D>;

    let pivots_ptr = stems_ptr.as_ptr().add(block_base_idx * 4) as *const f32;
    let mut pivots = [0.0f32; 8];
    std::ptr::copy_nonoverlapping(pivots_ptr, pivots.as_mut_ptr(), 8);

    let mut lower_vals = [0.0f32; 8];
    let mut upper_vals = [0.0f32; 8];
    for i in 0..8 {
        let (lower_offset, upper_offset) = super::child_interval_bounds_block3(i);
        lower_vals[i] = if lower_offset == 255 {
            f32::NEG_INFINITY
        } else {
            pivots[lower_offset as usize]
        };
        upper_vals[i] = if upper_offset == 255 {
            f32::INFINITY
        } else {
            pivots[upper_offset as usize]
        };
    }

    let lower = _mm256_loadu_ps(lower_vals.as_ptr());
    let upper = _mm256_loadu_ps(upper_vals.as_ptr());
    let query_vec = _mm256_set1_ps(query_wide);
    let old_off_abs_vec = _mm256_set1_ps(old_off.abs());
    let rd_vec = _mm256_set1_ps(rd);
    let best_dist_vec = _mm256_set1_ps(best_dist);
    let zero_vec = _mm256_setzero_ps();

    let below = _mm256_max_ps(_mm256_sub_ps(lower, query_vec), zero_vec);
    let above = _mm256_max_ps(_mm256_sub_ps(query_vec, upper), zero_vec);
    let interval = _mm256_add_ps(below, above);
    let rd_far = _mm256_add_ps(rd_vec, _mm256_sub_ps(interval, old_off_abs_vec));

    _mm256_movemask_ps(_mm256_cmp_ps(rd_far, best_dist_vec, _CMP_LE_OQ)) as u8
}

#[cfg(all(
    feature = "simd",
    target_arch = "x86_64",
    not(target_feature = "avx512f")
))]
#[inline(always)]
unsafe fn simd_backtrack_block4_f64_avx2_manhattan<
    A: Copy,
    D: DistanceMetricUnified<A, K, Output = f64>,
    const K: usize,
>(
    query_wide: f64,
    stems_ptr: NonNull<u8>,
    block_base_idx: usize,
    old_off: f64,
    rd: f64,
    best_dist: f64,
) -> u16 {
    use std::arch::x86_64::*;
    let _ = core::marker::PhantomData::<D>;

    let pivots_ptr = stems_ptr.as_ptr().add(block_base_idx * 8) as *const f64;
    let mut pivots = [0.0f64; 16];
    std::ptr::copy_nonoverlapping(pivots_ptr, pivots.as_mut_ptr(), 16);

    let mut lower_vals = [0.0f64; 16];
    let mut upper_vals = [0.0f64; 16];
    for i in 0..16 {
        let (lower_offset, upper_offset) = super::child_interval_bounds_block4(i);
        lower_vals[i] = if lower_offset == 255 {
            f64::NEG_INFINITY
        } else {
            pivots[lower_offset as usize]
        };
        upper_vals[i] = if upper_offset == 255 {
            f64::INFINITY
        } else {
            pivots[upper_offset as usize]
        };
    }

    let query_vec = _mm256_set1_pd(query_wide);
    let old_off_abs_vec = _mm256_set1_pd(old_off.abs());
    let rd_vec = _mm256_set1_pd(rd);
    let best_dist_vec = _mm256_set1_pd(best_dist);
    let zero_vec = _mm256_setzero_pd();

    let mut mask: u16 = 0;
    for chunk in 0..4 {
        let idx = chunk * 4;
        let lower = _mm256_loadu_pd(lower_vals.as_ptr().add(idx));
        let upper = _mm256_loadu_pd(upper_vals.as_ptr().add(idx));
        let below = _mm256_max_pd(_mm256_sub_pd(lower, query_vec), zero_vec);
        let above = _mm256_max_pd(_mm256_sub_pd(query_vec, upper), zero_vec);
        let interval = _mm256_add_pd(below, above);
        let rd_far = _mm256_add_pd(rd_vec, _mm256_sub_pd(interval, old_off_abs_vec));
        let chunk_mask = _mm256_movemask_pd(_mm256_cmp_pd(rd_far, best_dist_vec, _CMP_LE_OQ));
        mask |= (chunk_mask as u16) << (chunk * 4);
    }

    mask
}

#[cfg(all(
    feature = "simd",
    target_arch = "x86_64",
    not(target_feature = "avx512f")
))]
#[inline(always)]
unsafe fn simd_backtrack_block4_f32_avx2_manhattan<
    A: Copy,
    D: DistanceMetricUnified<A, K, Output = f32>,
    const K: usize,
>(
    query_wide: f32,
    stems_ptr: NonNull<u8>,
    block_base_idx: usize,
    old_off: f32,
    rd: f32,
    best_dist: f32,
) -> u16 {
    use std::arch::x86_64::*;
    let _ = core::marker::PhantomData::<D>;

    let pivots_ptr = stems_ptr.as_ptr().add(block_base_idx * 4) as *const f32;
    let mut pivots = [0.0f32; 16];
    std::ptr::copy_nonoverlapping(pivots_ptr, pivots.as_mut_ptr(), 16);

    let mut lower_vals = [0.0f32; 16];
    let mut upper_vals = [0.0f32; 16];
    for i in 0..16 {
        let (lower_offset, upper_offset) = super::child_interval_bounds_block4(i);
        lower_vals[i] = if lower_offset == 255 {
            f32::NEG_INFINITY
        } else {
            pivots[lower_offset as usize]
        };
        upper_vals[i] = if upper_offset == 255 {
            f32::INFINITY
        } else {
            pivots[upper_offset as usize]
        };
    }

    let query_vec = _mm256_set1_ps(query_wide);
    let old_off_abs_vec = _mm256_set1_ps(old_off.abs());
    let rd_vec = _mm256_set1_ps(rd);
    let best_dist_vec = _mm256_set1_ps(best_dist);
    let zero_vec = _mm256_setzero_ps();

    let lower_lo = _mm256_loadu_ps(lower_vals.as_ptr());
    let upper_lo = _mm256_loadu_ps(upper_vals.as_ptr());
    let below_lo = _mm256_max_ps(_mm256_sub_ps(lower_lo, query_vec), zero_vec);
    let above_lo = _mm256_max_ps(_mm256_sub_ps(query_vec, upper_lo), zero_vec);
    let interval_lo = _mm256_add_ps(below_lo, above_lo);
    let rd_far_lo = _mm256_add_ps(rd_vec, _mm256_sub_ps(interval_lo, old_off_abs_vec));
    let mask_lo = _mm256_movemask_ps(_mm256_cmp_ps(rd_far_lo, best_dist_vec, _CMP_LE_OQ)) as u16;

    let lower_hi = _mm256_loadu_ps(lower_vals.as_ptr().add(8));
    let upper_hi = _mm256_loadu_ps(upper_vals.as_ptr().add(8));
    let below_hi = _mm256_max_ps(_mm256_sub_ps(lower_hi, query_vec), zero_vec);
    let above_hi = _mm256_max_ps(_mm256_sub_ps(query_vec, upper_hi), zero_vec);
    let interval_hi = _mm256_add_ps(below_hi, above_hi);
    let rd_far_hi = _mm256_add_ps(rd_vec, _mm256_sub_ps(interval_hi, old_off_abs_vec));
    let mask_hi = _mm256_movemask_ps(_mm256_cmp_ps(rd_far_hi, best_dist_vec, _CMP_LE_OQ)) as u16;

    mask_lo | (mask_hi << 8)
}

// ====================================================================================
// x86_64 AVX-512 implementations
// ====================================================================================

#[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx512f"))]
#[inline(always)]
unsafe fn simd_backtrack_block3_f64_avx512_squared_euclidean<
    A: Copy,
    D: DistanceMetricUnified<A, K, Output = f64>,
    const K: usize,
>(
    query_wide: f64,
    stems_ptr: NonNull<u8>,
    block_base_idx: usize,
    old_off: f64,
    rd: f64,
    best_dist: f64,
) -> u8 {
    use std::arch::x86_64::*;

    // Load 8 pivots into a scalar array
    let ptr = stems_ptr.as_ptr().add(block_base_idx * 8) as *const f64;
    let mut pivots = [0.0f64; 8];
    std::ptr::copy_nonoverlapping(ptr, pivots.as_mut_ptr(), 8);

    // Precompute child-indexed lower/upper bounds
    let mut lower_vals = [0.0f64; 8];
    let mut upper_vals = [0.0f64; 8];
    for i in 0..8 {
        let (lower_offset, upper_offset) = super::child_interval_bounds_block3(i);
        lower_vals[i] = if lower_offset == 255 {
            f64::NEG_INFINITY
        } else {
            pivots[lower_offset as usize]
        };
        upper_vals[i] = if upper_offset == 255 {
            f64::INFINITY
        } else {
            pivots[upper_offset as usize]
        };
    }

    // All 8 children fit in a single __m512d
    let lower = _mm512_loadu_pd(lower_vals.as_ptr());
    let upper = _mm512_loadu_pd(upper_vals.as_ptr());

    let query_vec = _mm512_set1_pd(query_wide);
    let old_off_sq_vec = _mm512_set1_pd(old_off * old_off);
    let rd_vec = _mm512_set1_pd(rd);
    let best_dist_vec = _mm512_set1_pd(best_dist);
    let zero_vec = _mm512_setzero_pd();

    // Interval distance: max(0, lower - query) + max(0, query - upper)
    let below = _mm512_max_pd(_mm512_sub_pd(lower, query_vec), zero_vec);
    let above = _mm512_max_pd(_mm512_sub_pd(query_vec, upper), zero_vec);
    let interval = _mm512_add_pd(below, above);

    // new_sq = interval² (SquaredEuclidean)
    let new_sq = _mm512_mul_pd(interval, interval);

    // rd_far = rd - old_off² + new_off²
    let rd_far = _mm512_add_pd(rd_vec, _mm512_sub_pd(new_sq, old_off_sq_vec));

    // Compare rd_far <= best_dist → __mmask8 directly
    _mm512_cmp_pd_mask(rd_far, best_dist_vec, _CMP_LE_OQ)
}

#[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx512f"))]
#[inline(always)]
unsafe fn simd_backtrack_block3_f32_avx512_squared_euclidean<
    A: Copy,
    D: DistanceMetricUnified<A, K, Output = f32>,
    const K: usize,
>(
    query_wide: f32,
    stems_ptr: NonNull<u8>,
    block_base_idx: usize,
    old_off: f32,
    rd: f32,
    best_dist: f32,
) -> u8 {
    use std::arch::x86_64::*;

    // Load 8 pivots into a scalar array
    let ptr = stems_ptr.as_ptr().add(block_base_idx * 4) as *const f32;
    let mut pivots = [0.0f32; 8];
    std::ptr::copy_nonoverlapping(ptr, pivots.as_mut_ptr(), 8);

    // Precompute child-indexed lower/upper bounds
    let mut lower_vals = [0.0f32; 8];
    let mut upper_vals = [0.0f32; 8];
    for i in 0..8 {
        let (lower_offset, upper_offset) = super::child_interval_bounds_block3(i);
        lower_vals[i] = if lower_offset == 255 {
            f32::NEG_INFINITY
        } else {
            pivots[lower_offset as usize]
        };
        upper_vals[i] = if upper_offset == 255 {
            f32::INFINITY
        } else {
            pivots[upper_offset as usize]
        };
    }

    // 8 children fit in __m256 (AVX-512 implies AVX2 availability)
    let lower = _mm256_loadu_ps(lower_vals.as_ptr());
    let upper = _mm256_loadu_ps(upper_vals.as_ptr());

    let query_vec = _mm256_set1_ps(query_wide);
    let old_off_sq_vec = _mm256_set1_ps(old_off * old_off);
    let rd_vec = _mm256_set1_ps(rd);
    let best_dist_vec = _mm256_set1_ps(best_dist);
    let zero_vec = _mm256_setzero_ps();

    // Interval distance: max(0, lower - query) + max(0, query - upper)
    let below = _mm256_max_ps(_mm256_sub_ps(lower, query_vec), zero_vec);
    let above = _mm256_max_ps(_mm256_sub_ps(query_vec, upper), zero_vec);
    let interval = _mm256_add_ps(below, above);

    // new_sq = interval² (SquaredEuclidean)
    let new_sq = _mm256_mul_ps(interval, interval);

    // rd_far = rd - old_off² + new_off²
    let rd_far = _mm256_add_ps(rd_vec, _mm256_sub_ps(new_sq, old_off_sq_vec));

    // Compare rd_far <= best_dist
    let cmp = _mm256_cmp_ps(rd_far, best_dist_vec, _CMP_LE_OQ);
    _mm256_movemask_ps(cmp) as u8
}

#[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx512f"))]
#[inline(always)]
unsafe fn simd_backtrack_block4_f64_avx512_squared_euclidean<
    A: Copy,
    D: DistanceMetricUnified<A, K, Output = f64>,
    const K: usize,
>(
    query_wide: f64,
    stems_ptr: NonNull<u8>,
    block_base_idx: usize,
    old_off: f64,
    rd: f64,
    best_dist: f64,
) -> u16 {
    use std::arch::x86_64::*;

    // Load 16 pivots into a scalar array
    let ptr = stems_ptr.as_ptr().add(block_base_idx * 8) as *const f64;
    let mut pivots = [0.0f64; 16];
    std::ptr::copy_nonoverlapping(ptr, pivots.as_mut_ptr(), 16);

    // Precompute child-indexed lower/upper bounds
    let mut lower_vals = [0.0f64; 16];
    let mut upper_vals = [0.0f64; 16];
    for i in 0..16 {
        let (lower_offset, upper_offset) = super::child_interval_bounds_block4(i);
        lower_vals[i] = if lower_offset == 255 {
            f64::NEG_INFINITY
        } else {
            pivots[lower_offset as usize]
        };
        upper_vals[i] = if upper_offset == 255 {
            f64::INFINITY
        } else {
            pivots[upper_offset as usize]
        };
    }

    let query_vec = _mm512_set1_pd(query_wide);
    let old_off_sq_vec = _mm512_set1_pd(old_off * old_off);
    let rd_vec = _mm512_set1_pd(rd);
    let best_dist_vec = _mm512_set1_pd(best_dist);
    let zero_vec = _mm512_setzero_pd();

    // Children 0-7
    let lower_lo = _mm512_loadu_pd(lower_vals.as_ptr());
    let upper_lo = _mm512_loadu_pd(upper_vals.as_ptr());

    let below_lo = _mm512_max_pd(_mm512_sub_pd(lower_lo, query_vec), zero_vec);
    let above_lo = _mm512_max_pd(_mm512_sub_pd(query_vec, upper_lo), zero_vec);
    let interval_lo = _mm512_add_pd(below_lo, above_lo);
    let new_sq_lo = _mm512_mul_pd(interval_lo, interval_lo);
    let rd_far_lo = _mm512_add_pd(rd_vec, _mm512_sub_pd(new_sq_lo, old_off_sq_vec));
    let mask_lo = _mm512_cmp_pd_mask(rd_far_lo, best_dist_vec, _CMP_LE_OQ);

    // Children 8-15
    let lower_hi = _mm512_loadu_pd(lower_vals.as_ptr().add(8));
    let upper_hi = _mm512_loadu_pd(upper_vals.as_ptr().add(8));

    let below_hi = _mm512_max_pd(_mm512_sub_pd(lower_hi, query_vec), zero_vec);
    let above_hi = _mm512_max_pd(_mm512_sub_pd(query_vec, upper_hi), zero_vec);
    let interval_hi = _mm512_add_pd(below_hi, above_hi);
    let new_sq_hi = _mm512_mul_pd(interval_hi, interval_hi);
    let rd_far_hi = _mm512_add_pd(rd_vec, _mm512_sub_pd(new_sq_hi, old_off_sq_vec));
    let mask_hi = _mm512_cmp_pd_mask(rd_far_hi, best_dist_vec, _CMP_LE_OQ);

    // Combine: low 8 bits | high 8 bits
    (mask_lo as u16) | ((mask_hi as u16) << 8)
}

#[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx512f"))]
#[inline(always)]
unsafe fn simd_backtrack_block4_f32_avx512_squared_euclidean<
    A: Copy,
    D: DistanceMetricUnified<A, K, Output = f32>,
    const K: usize,
>(
    query_wide: f32,
    stems_ptr: NonNull<u8>,
    block_base_idx: usize,
    old_off: f32,
    rd: f32,
    best_dist: f32,
) -> u16 {
    use std::arch::x86_64::*;

    // Load 16 pivots into a scalar array
    let ptr = stems_ptr.as_ptr().add(block_base_idx * 4) as *const f32;
    let mut pivots = [0.0f32; 16];
    std::ptr::copy_nonoverlapping(ptr, pivots.as_mut_ptr(), 16);

    // Precompute child-indexed lower/upper bounds
    let mut lower_vals = [0.0f32; 16];
    let mut upper_vals = [0.0f32; 16];
    for i in 0..16 {
        let (lower_offset, upper_offset) = super::child_interval_bounds_block4(i);
        lower_vals[i] = if lower_offset == 255 {
            f32::NEG_INFINITY
        } else {
            pivots[lower_offset as usize]
        };
        upper_vals[i] = if upper_offset == 255 {
            f32::INFINITY
        } else {
            pivots[upper_offset as usize]
        };
    }

    // All 16 children fit in a single __m512
    let lower = _mm512_loadu_ps(lower_vals.as_ptr());
    let upper = _mm512_loadu_ps(upper_vals.as_ptr());

    let query_vec = _mm512_set1_ps(query_wide);
    let old_off_sq_vec = _mm512_set1_ps(old_off * old_off);
    let rd_vec = _mm512_set1_ps(rd);
    let best_dist_vec = _mm512_set1_ps(best_dist);
    let zero_vec = _mm512_setzero_ps();

    // Interval distance: max(0, lower - query) + max(0, query - upper)
    let below = _mm512_max_ps(_mm512_sub_ps(lower, query_vec), zero_vec);
    let above = _mm512_max_ps(_mm512_sub_ps(query_vec, upper), zero_vec);
    let interval = _mm512_add_ps(below, above);

    // new_sq = interval² (SquaredEuclidean)
    let new_sq = _mm512_mul_ps(interval, interval);

    // rd_far = rd - old_off² + new_off²
    let rd_far = _mm512_add_ps(rd_vec, _mm512_sub_ps(new_sq, old_off_sq_vec));

    // Compare rd_far <= best_dist → __mmask16 directly
    _mm512_cmp_ps_mask(rd_far, best_dist_vec, _CMP_LE_OQ)
}

#[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx512f"))]
#[inline(always)]
unsafe fn simd_backtrack_block3_f64_avx512_manhattan<
    A: Copy,
    D: DistanceMetricUnified<A, K, Output = f64>,
    const K: usize,
>(
    query_wide: f64,
    stems_ptr: NonNull<u8>,
    block_base_idx: usize,
    old_off: f64,
    rd: f64,
    best_dist: f64,
) -> u8 {
    use std::arch::x86_64::*;
    let _ = core::marker::PhantomData::<D>;

    let ptr = stems_ptr.as_ptr().add(block_base_idx * 8) as *const f64;
    let mut pivots = [0.0f64; 8];
    std::ptr::copy_nonoverlapping(ptr, pivots.as_mut_ptr(), 8);

    let mut lower_vals = [0.0f64; 8];
    let mut upper_vals = [0.0f64; 8];
    for i in 0..8 {
        let (lower_offset, upper_offset) = super::child_interval_bounds_block3(i);
        lower_vals[i] = if lower_offset == 255 {
            f64::NEG_INFINITY
        } else {
            pivots[lower_offset as usize]
        };
        upper_vals[i] = if upper_offset == 255 {
            f64::INFINITY
        } else {
            pivots[upper_offset as usize]
        };
    }

    let lower = _mm512_loadu_pd(lower_vals.as_ptr());
    let upper = _mm512_loadu_pd(upper_vals.as_ptr());
    let query_vec = _mm512_set1_pd(query_wide);
    let old_off_abs_vec = _mm512_set1_pd(old_off.abs());
    let rd_vec = _mm512_set1_pd(rd);
    let best_dist_vec = _mm512_set1_pd(best_dist);
    let zero_vec = _mm512_setzero_pd();

    let below = _mm512_max_pd(_mm512_sub_pd(lower, query_vec), zero_vec);
    let above = _mm512_max_pd(_mm512_sub_pd(query_vec, upper), zero_vec);
    let interval = _mm512_add_pd(below, above);
    let rd_far = _mm512_add_pd(rd_vec, _mm512_sub_pd(interval, old_off_abs_vec));

    _mm512_cmp_pd_mask(rd_far, best_dist_vec, _CMP_LE_OQ)
}

#[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx512f"))]
#[inline(always)]
unsafe fn simd_backtrack_block3_f32_avx512_manhattan<
    A: Copy,
    D: DistanceMetricUnified<A, K, Output = f32>,
    const K: usize,
>(
    query_wide: f32,
    stems_ptr: NonNull<u8>,
    block_base_idx: usize,
    old_off: f32,
    rd: f32,
    best_dist: f32,
) -> u8 {
    use std::arch::x86_64::*;
    let _ = core::marker::PhantomData::<D>;

    let ptr = stems_ptr.as_ptr().add(block_base_idx * 4) as *const f32;
    let mut pivots = [0.0f32; 8];
    std::ptr::copy_nonoverlapping(ptr, pivots.as_mut_ptr(), 8);

    let mut lower_vals = [0.0f32; 8];
    let mut upper_vals = [0.0f32; 8];
    for i in 0..8 {
        let (lower_offset, upper_offset) = super::child_interval_bounds_block3(i);
        lower_vals[i] = if lower_offset == 255 {
            f32::NEG_INFINITY
        } else {
            pivots[lower_offset as usize]
        };
        upper_vals[i] = if upper_offset == 255 {
            f32::INFINITY
        } else {
            pivots[upper_offset as usize]
        };
    }

    let lower = _mm256_loadu_ps(lower_vals.as_ptr());
    let upper = _mm256_loadu_ps(upper_vals.as_ptr());
    let query_vec = _mm256_set1_ps(query_wide);
    let old_off_abs_vec = _mm256_set1_ps(old_off.abs());
    let rd_vec = _mm256_set1_ps(rd);
    let best_dist_vec = _mm256_set1_ps(best_dist);
    let zero_vec = _mm256_setzero_ps();

    let below = _mm256_max_ps(_mm256_sub_ps(lower, query_vec), zero_vec);
    let above = _mm256_max_ps(_mm256_sub_ps(query_vec, upper), zero_vec);
    let interval = _mm256_add_ps(below, above);
    let rd_far = _mm256_add_ps(rd_vec, _mm256_sub_ps(interval, old_off_abs_vec));

    _mm256_movemask_ps(_mm256_cmp_ps(rd_far, best_dist_vec, _CMP_LE_OQ)) as u8
}

#[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx512f"))]
#[inline(always)]
unsafe fn simd_backtrack_block4_f64_avx512_manhattan<
    A: Copy,
    D: DistanceMetricUnified<A, K, Output = f64>,
    const K: usize,
>(
    query_wide: f64,
    stems_ptr: NonNull<u8>,
    block_base_idx: usize,
    old_off: f64,
    rd: f64,
    best_dist: f64,
) -> u16 {
    use std::arch::x86_64::*;
    let _ = core::marker::PhantomData::<D>;

    let ptr = stems_ptr.as_ptr().add(block_base_idx * 8) as *const f64;
    let mut pivots = [0.0f64; 16];
    std::ptr::copy_nonoverlapping(ptr, pivots.as_mut_ptr(), 16);

    let mut lower_vals = [0.0f64; 16];
    let mut upper_vals = [0.0f64; 16];
    for i in 0..16 {
        let (lower_offset, upper_offset) = super::child_interval_bounds_block4(i);
        lower_vals[i] = if lower_offset == 255 {
            f64::NEG_INFINITY
        } else {
            pivots[lower_offset as usize]
        };
        upper_vals[i] = if upper_offset == 255 {
            f64::INFINITY
        } else {
            pivots[upper_offset as usize]
        };
    }

    let query_vec = _mm512_set1_pd(query_wide);
    let old_off_abs_vec = _mm512_set1_pd(old_off.abs());
    let rd_vec = _mm512_set1_pd(rd);
    let best_dist_vec = _mm512_set1_pd(best_dist);
    let zero_vec = _mm512_setzero_pd();

    let lower_lo = _mm512_loadu_pd(lower_vals.as_ptr());
    let upper_lo = _mm512_loadu_pd(upper_vals.as_ptr());
    let below_lo = _mm512_max_pd(_mm512_sub_pd(lower_lo, query_vec), zero_vec);
    let above_lo = _mm512_max_pd(_mm512_sub_pd(query_vec, upper_lo), zero_vec);
    let interval_lo = _mm512_add_pd(below_lo, above_lo);
    let rd_far_lo = _mm512_add_pd(rd_vec, _mm512_sub_pd(interval_lo, old_off_abs_vec));
    let mask_lo = _mm512_cmp_pd_mask(rd_far_lo, best_dist_vec, _CMP_LE_OQ);

    let lower_hi = _mm512_loadu_pd(lower_vals.as_ptr().add(8));
    let upper_hi = _mm512_loadu_pd(upper_vals.as_ptr().add(8));
    let below_hi = _mm512_max_pd(_mm512_sub_pd(lower_hi, query_vec), zero_vec);
    let above_hi = _mm512_max_pd(_mm512_sub_pd(query_vec, upper_hi), zero_vec);
    let interval_hi = _mm512_add_pd(below_hi, above_hi);
    let rd_far_hi = _mm512_add_pd(rd_vec, _mm512_sub_pd(interval_hi, old_off_abs_vec));
    let mask_hi = _mm512_cmp_pd_mask(rd_far_hi, best_dist_vec, _CMP_LE_OQ);

    (mask_lo as u16) | ((mask_hi as u16) << 8)
}

#[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx512f"))]
#[inline(always)]
unsafe fn simd_backtrack_block4_f32_avx512_manhattan<
    A: Copy,
    D: DistanceMetricUnified<A, K, Output = f32>,
    const K: usize,
>(
    query_wide: f32,
    stems_ptr: NonNull<u8>,
    block_base_idx: usize,
    old_off: f32,
    rd: f32,
    best_dist: f32,
) -> u16 {
    use std::arch::x86_64::*;
    let _ = core::marker::PhantomData::<D>;

    let ptr = stems_ptr.as_ptr().add(block_base_idx * 4) as *const f32;
    let mut pivots = [0.0f32; 16];
    std::ptr::copy_nonoverlapping(ptr, pivots.as_mut_ptr(), 16);

    let mut lower_vals = [0.0f32; 16];
    let mut upper_vals = [0.0f32; 16];
    for i in 0..16 {
        let (lower_offset, upper_offset) = super::child_interval_bounds_block4(i);
        lower_vals[i] = if lower_offset == 255 {
            f32::NEG_INFINITY
        } else {
            pivots[lower_offset as usize]
        };
        upper_vals[i] = if upper_offset == 255 {
            f32::INFINITY
        } else {
            pivots[upper_offset as usize]
        };
    }

    let lower = _mm512_loadu_ps(lower_vals.as_ptr());
    let upper = _mm512_loadu_ps(upper_vals.as_ptr());
    let query_vec = _mm512_set1_ps(query_wide);
    let old_off_abs_vec = _mm512_set1_ps(old_off.abs());
    let rd_vec = _mm512_set1_ps(rd);
    let best_dist_vec = _mm512_set1_ps(best_dist);
    let zero_vec = _mm512_setzero_ps();

    let below = _mm512_max_ps(_mm512_sub_ps(lower, query_vec), zero_vec);
    let above = _mm512_max_ps(_mm512_sub_ps(query_vec, upper), zero_vec);
    let interval = _mm512_add_ps(below, above);
    let rd_far = _mm512_add_ps(rd_vec, _mm512_sub_ps(interval, old_off_abs_vec));

    _mm512_cmp_ps_mask(rd_far, best_dist_vec, _CMP_LE_OQ)
}

// ====================================================================================
// aarch64 NEON implementations
// ====================================================================================

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
#[inline(always)]
#[allow(dead_code)]
unsafe fn simd_backtrack_block3_f64_neon_squared_euclidean<
    A: Copy,
    D: DistanceMetricUnified<A, K, Output = f64>,
    const K: usize,
>(
    query_wide: f64,
    stems_ptr: NonNull<u8>,
    block_base_idx: usize,
    old_off: f64,
    rd: f64,
    best_dist: f64,
) -> u8 {
    use core::arch::aarch64::*;
    let _ = core::marker::PhantomData::<D>;

    let pivots_ptr = stems_ptr.as_ptr().add(block_base_idx * 8) as *const f64;
    let mut pivots = [0.0f64; 8];
    core::ptr::copy_nonoverlapping(pivots_ptr, pivots.as_mut_ptr(), 8);

    let mut lower_vals = [0.0f64; 8];
    let mut upper_vals = [0.0f64; 8];

    for i in 0..8 {
        let (lower_offset, upper_offset) = super::child_interval_bounds_block3(i);
        lower_vals[i] = if lower_offset == 255 {
            f64::NEG_INFINITY
        } else {
            pivots[lower_offset as usize]
        };
        upper_vals[i] = if upper_offset == 255 {
            f64::INFINITY
        } else {
            pivots[upper_offset as usize]
        };
    }

    let query_vec = vdupq_n_f64(query_wide);
    let old_off_sq_vec = vmulq_f64(vdupq_n_f64(old_off), vdupq_n_f64(old_off));
    let rd_vec = vdupq_n_f64(rd);
    let best_dist_vec = vdupq_n_f64(best_dist);
    let zero_vec = vdupq_n_f64(0.0);

    let mut mask: u8 = 0;

    for chunk in 0..4 {
        let idx = chunk * 2;
        let lower = vld1q_f64(lower_vals.as_ptr().add(idx));
        let upper = vld1q_f64(upper_vals.as_ptr().add(idx));

        let below = vmaxq_f64(vsubq_f64(lower, query_vec), zero_vec);
        let above = vmaxq_f64(vsubq_f64(query_vec, upper), zero_vec);
        let interval = vaddq_f64(below, above);

        let new_off_sq = vmulq_f64(interval, interval);
        let delta = vsubq_f64(new_off_sq, old_off_sq_vec);
        let rd_far = vaddq_f64(rd_vec, delta);

        let cmp = vcleq_f64(rd_far, best_dist_vec);

        let weights = match chunk {
            0 => [1u64, 2u64],
            1 => [4u64, 8u64],
            2 => [16u64, 32u64],
            _ => [64u64, 128u64],
        };

        let mask_chunk = vaddvq_u64(vandq_u64(cmp, vld1q_u64(weights.as_ptr())));
        mask |= mask_chunk as u8;
    }

    mask
}

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
#[inline(always)]
#[allow(dead_code)]
unsafe fn simd_backtrack_block3_f32_neon_squared_euclidean<
    A: Copy,
    D: DistanceMetricUnified<A, K, Output = f32>,
    const K: usize,
>(
    query_wide: f32,
    stems_ptr: NonNull<u8>,
    block_base_idx: usize,
    old_off: f32,
    rd: f32,
    best_dist: f32,
) -> u8 {
    use core::arch::aarch64::*;
    let _ = core::marker::PhantomData::<D>;

    let pivots_ptr = stems_ptr.as_ptr().add(block_base_idx * 4) as *const f32;
    let mut pivots = [0.0f32; 8];
    core::ptr::copy_nonoverlapping(pivots_ptr, pivots.as_mut_ptr(), 8);

    let mut lower_vals = [0.0f32; 8];
    let mut upper_vals = [0.0f32; 8];

    for i in 0..8 {
        let (lower_offset, upper_offset) = super::child_interval_bounds_block3(i);
        lower_vals[i] = if lower_offset == 255 {
            f32::NEG_INFINITY
        } else {
            pivots[lower_offset as usize]
        };
        upper_vals[i] = if upper_offset == 255 {
            f32::INFINITY
        } else {
            pivots[upper_offset as usize]
        };
    }

    let query_vec = vdupq_n_f32(query_wide);
    let old_off_sq_vec = vmulq_f32(vdupq_n_f32(old_off), vdupq_n_f32(old_off));
    let rd_vec = vdupq_n_f32(rd);
    let best_dist_vec = vdupq_n_f32(best_dist);
    let zero_vec = vdupq_n_f32(0.0);

    let mut mask: u8 = 0;

    for chunk in 0..2 {
        let idx = chunk * 4;
        let lower = vld1q_f32(lower_vals.as_ptr().add(idx));
        let upper = vld1q_f32(upper_vals.as_ptr().add(idx));

        let below = vmaxq_f32(vsubq_f32(lower, query_vec), zero_vec);
        let above = vmaxq_f32(vsubq_f32(query_vec, upper), zero_vec);
        let interval = vaddq_f32(below, above);

        let new_off_sq = vmulq_f32(interval, interval);
        let delta = vsubq_f32(new_off_sq, old_off_sq_vec);
        let rd_far = vaddq_f32(rd_vec, delta);

        let cmp = vcleq_f32(rd_far, best_dist_vec);

        let weights = match chunk {
            0 => [1u32, 2u32, 4u32, 8u32],
            _ => [16u32, 32u32, 64u32, 128u32],
        };

        let mask_chunk = vaddvq_u32(vandq_u32(cmp, vld1q_u32(weights.as_ptr())));
        mask |= mask_chunk as u8;
    }

    mask
}

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
#[inline(always)]
#[allow(dead_code)]
unsafe fn simd_backtrack_block4_f64_neon_squared_euclidean<
    A: Copy,
    D: DistanceMetricUnified<A, K, Output = f64>,
    const K: usize,
>(
    query_wide: f64,
    stems_ptr: NonNull<u8>,
    block_base_idx: usize,
    old_off: f64,
    rd: f64,
    best_dist: f64,
) -> u16 {
    use core::arch::aarch64::*;
    let _ = core::marker::PhantomData::<D>;

    let pivots_ptr = stems_ptr.as_ptr().add(block_base_idx * 8) as *const f64;
    let mut pivots = [0.0f64; 16];
    core::ptr::copy_nonoverlapping(pivots_ptr, pivots.as_mut_ptr(), 16);

    let mut lower_vals = [0.0f64; 16];
    let mut upper_vals = [0.0f64; 16];

    for i in 0..16 {
        let (lower_offset, upper_offset) = super::child_interval_bounds_block4(i);
        lower_vals[i] = if lower_offset == 255 {
            f64::NEG_INFINITY
        } else {
            pivots[lower_offset as usize]
        };
        upper_vals[i] = if upper_offset == 255 {
            f64::INFINITY
        } else {
            pivots[upper_offset as usize]
        };
    }

    let query_vec = vdupq_n_f64(query_wide);
    let old_off_sq_vec = vmulq_f64(vdupq_n_f64(old_off), vdupq_n_f64(old_off));
    let rd_vec = vdupq_n_f64(rd);
    let best_dist_vec = vdupq_n_f64(best_dist);
    let zero_vec = vdupq_n_f64(0.0);

    let mut mask: u16 = 0;

    for chunk in 0..8 {
        let idx = chunk * 2;
        let lower = vld1q_f64(lower_vals.as_ptr().add(idx));
        let upper = vld1q_f64(upper_vals.as_ptr().add(idx));

        let below = vmaxq_f64(vsubq_f64(lower, query_vec), zero_vec);
        let above = vmaxq_f64(vsubq_f64(query_vec, upper), zero_vec);
        let interval = vaddq_f64(below, above);

        let new_off_sq = vmulq_f64(interval, interval);
        let delta = vsubq_f64(new_off_sq, old_off_sq_vec);
        let rd_far = vaddq_f64(rd_vec, delta);

        let cmp = vcleq_f64(rd_far, best_dist_vec);

        let weights = match chunk {
            0 => [1u64, 2u64],
            1 => [4u64, 8u64],
            2 => [16u64, 32u64],
            3 => [64u64, 128u64],
            4 => [256u64, 512u64],
            5 => [1024u64, 2048u64],
            6 => [4096u64, 8192u64],
            _ => [16384u64, 32768u64],
        };

        let mask_chunk = vaddvq_u64(vandq_u64(cmp, vld1q_u64(weights.as_ptr())));
        mask |= mask_chunk as u16;
    }

    mask
}

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
#[inline(always)]
#[allow(dead_code)]
unsafe fn simd_backtrack_block4_f32_neon_squared_euclidean<
    A: Copy,
    D: DistanceMetricUnified<A, K, Output = f32>,
    const K: usize,
>(
    query_wide: f32,
    stems_ptr: NonNull<u8>,
    block_base_idx: usize,
    old_off: f32,
    rd: f32,
    best_dist: f32,
) -> u16 {
    use core::arch::aarch64::*;
    let _ = core::marker::PhantomData::<D>;

    let pivots_ptr = stems_ptr.as_ptr().add(block_base_idx * 4) as *const f32;
    let mut pivots = [0.0f32; 16];
    core::ptr::copy_nonoverlapping(pivots_ptr, pivots.as_mut_ptr(), 16);

    let mut lower_vals = [0.0f32; 16];
    let mut upper_vals = [0.0f32; 16];

    for i in 0..16 {
        let (lower_offset, upper_offset) = super::child_interval_bounds_block4(i);
        lower_vals[i] = if lower_offset == 255 {
            f32::NEG_INFINITY
        } else {
            pivots[lower_offset as usize]
        };
        upper_vals[i] = if upper_offset == 255 {
            f32::INFINITY
        } else {
            pivots[upper_offset as usize]
        };
    }

    let query_vec = vdupq_n_f32(query_wide);
    let old_off_sq_vec = vmulq_f32(vdupq_n_f32(old_off), vdupq_n_f32(old_off));
    let rd_vec = vdupq_n_f32(rd);
    let best_dist_vec = vdupq_n_f32(best_dist);
    let zero_vec = vdupq_n_f32(0.0);

    let mut mask: u16 = 0;

    for chunk in 0..4 {
        let idx = chunk * 4;
        let lower = vld1q_f32(lower_vals.as_ptr().add(idx));
        let upper = vld1q_f32(upper_vals.as_ptr().add(idx));

        let below = vmaxq_f32(vsubq_f32(lower, query_vec), zero_vec);
        let above = vmaxq_f32(vsubq_f32(query_vec, upper), zero_vec);
        let interval = vaddq_f32(below, above);

        let new_off_sq = vmulq_f32(interval, interval);
        let delta = vsubq_f32(new_off_sq, old_off_sq_vec);
        let rd_far = vaddq_f32(rd_vec, delta);

        let cmp = vcleq_f32(rd_far, best_dist_vec);

        let weights = match chunk {
            0 => [1u32, 2u32, 4u32, 8u32],
            1 => [16u32, 32u32, 64u32, 128u32],
            2 => [256u32, 512u32, 1024u32, 2048u32],
            _ => [4096u32, 8192u32, 16384u32, 32768u32],
        };

        let mask_chunk = vaddvq_u32(vandq_u32(cmp, vld1q_u32(weights.as_ptr())));
        mask |= mask_chunk as u16;
    }

    mask
}

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
#[inline(always)]
#[allow(dead_code)]
unsafe fn simd_backtrack_block3_f64_neon_manhattan<
    A: Copy,
    D: DistanceMetricUnified<A, K, Output = f64>,
    const K: usize,
>(
    query_wide: f64,
    stems_ptr: NonNull<u8>,
    block_base_idx: usize,
    old_off: f64,
    rd: f64,
    best_dist: f64,
) -> u8 {
    use core::arch::aarch64::*;
    let _ = core::marker::PhantomData::<D>;

    let pivots_ptr = stems_ptr.as_ptr().add(block_base_idx * 8) as *const f64;
    let mut pivots = [0.0f64; 8];
    core::ptr::copy_nonoverlapping(pivots_ptr, pivots.as_mut_ptr(), 8);

    let mut lower_vals = [0.0f64; 8];
    let mut upper_vals = [0.0f64; 8];

    for i in 0..8 {
        let (lower_offset, upper_offset) = super::child_interval_bounds_block3(i);
        lower_vals[i] = if lower_offset == 255 {
            f64::NEG_INFINITY
        } else {
            pivots[lower_offset as usize]
        };
        upper_vals[i] = if upper_offset == 255 {
            f64::INFINITY
        } else {
            pivots[upper_offset as usize]
        };
    }

    let query_vec = vdupq_n_f64(query_wide);
    let old_off_abs_vec = vabsq_f64(vdupq_n_f64(old_off));
    let rd_vec = vdupq_n_f64(rd);
    let best_dist_vec = vdupq_n_f64(best_dist);
    let zero_vec = vdupq_n_f64(0.0);

    let mut mask: u8 = 0;

    for chunk in 0..4 {
        let idx = chunk * 2;
        let lower = vld1q_f64(lower_vals.as_ptr().add(idx));
        let upper = vld1q_f64(upper_vals.as_ptr().add(idx));

        let below = vmaxq_f64(vsubq_f64(lower, query_vec), zero_vec);
        let above = vmaxq_f64(vsubq_f64(query_vec, upper), zero_vec);
        let interval = vaddq_f64(below, above);

        let delta = vsubq_f64(interval, old_off_abs_vec);
        let rd_far = vaddq_f64(rd_vec, delta);

        let cmp = vcleq_f64(rd_far, best_dist_vec);

        let weights = match chunk {
            0 => [1u64, 2u64],
            1 => [4u64, 8u64],
            2 => [16u64, 32u64],
            _ => [64u64, 128u64],
        };

        let mask_chunk = vaddvq_u64(vandq_u64(cmp, vld1q_u64(weights.as_ptr())));
        mask |= mask_chunk as u8;
    }

    mask
}

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
#[inline(always)]
#[allow(dead_code)]
unsafe fn simd_backtrack_block3_f32_neon_manhattan<
    A: Copy,
    D: DistanceMetricUnified<A, K, Output = f32>,
    const K: usize,
>(
    query_wide: f32,
    stems_ptr: NonNull<u8>,
    block_base_idx: usize,
    old_off: f32,
    rd: f32,
    best_dist: f32,
) -> u8 {
    use core::arch::aarch64::*;
    let _ = core::marker::PhantomData::<D>;

    let pivots_ptr = stems_ptr.as_ptr().add(block_base_idx * 4) as *const f32;
    let mut pivots = [0.0f32; 8];
    core::ptr::copy_nonoverlapping(pivots_ptr, pivots.as_mut_ptr(), 8);

    let mut lower_vals = [0.0f32; 8];
    let mut upper_vals = [0.0f32; 8];

    for i in 0..8 {
        let (lower_offset, upper_offset) = super::child_interval_bounds_block3(i);
        lower_vals[i] = if lower_offset == 255 {
            f32::NEG_INFINITY
        } else {
            pivots[lower_offset as usize]
        };
        upper_vals[i] = if upper_offset == 255 {
            f32::INFINITY
        } else {
            pivots[upper_offset as usize]
        };
    }

    let query_vec = vdupq_n_f32(query_wide);
    let old_off_abs_vec = vabsq_f32(vdupq_n_f32(old_off));
    let rd_vec = vdupq_n_f32(rd);
    let best_dist_vec = vdupq_n_f32(best_dist);
    let zero_vec = vdupq_n_f32(0.0);

    let mut mask: u8 = 0;

    for chunk in 0..2 {
        let idx = chunk * 4;
        let lower = vld1q_f32(lower_vals.as_ptr().add(idx));
        let upper = vld1q_f32(upper_vals.as_ptr().add(idx));

        let below = vmaxq_f32(vsubq_f32(lower, query_vec), zero_vec);
        let above = vmaxq_f32(vsubq_f32(query_vec, upper), zero_vec);
        let interval = vaddq_f32(below, above);

        let delta = vsubq_f32(interval, old_off_abs_vec);
        let rd_far = vaddq_f32(rd_vec, delta);

        let cmp = vcleq_f32(rd_far, best_dist_vec);

        let weights = match chunk {
            0 => [1u32, 2u32, 4u32, 8u32],
            _ => [16u32, 32u32, 64u32, 128u32],
        };

        let mask_chunk = vaddvq_u32(vandq_u32(cmp, vld1q_u32(weights.as_ptr())));
        mask |= mask_chunk as u8;
    }

    mask
}

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
#[inline(always)]
#[allow(dead_code)]
unsafe fn simd_backtrack_block4_f64_neon_manhattan<
    A: Copy,
    D: DistanceMetricUnified<A, K, Output = f64>,
    const K: usize,
>(
    query_wide: f64,
    stems_ptr: NonNull<u8>,
    block_base_idx: usize,
    old_off: f64,
    rd: f64,
    best_dist: f64,
) -> u16 {
    use core::arch::aarch64::*;
    let _ = core::marker::PhantomData::<D>;

    let pivots_ptr = stems_ptr.as_ptr().add(block_base_idx * 8) as *const f64;
    let mut pivots = [0.0f64; 16];
    core::ptr::copy_nonoverlapping(pivots_ptr, pivots.as_mut_ptr(), 16);

    let mut lower_vals = [0.0f64; 16];
    let mut upper_vals = [0.0f64; 16];

    for i in 0..16 {
        let (lower_offset, upper_offset) = super::child_interval_bounds_block4(i);
        lower_vals[i] = if lower_offset == 255 {
            f64::NEG_INFINITY
        } else {
            pivots[lower_offset as usize]
        };
        upper_vals[i] = if upper_offset == 255 {
            f64::INFINITY
        } else {
            pivots[upper_offset as usize]
        };
    }

    let query_vec = vdupq_n_f64(query_wide);
    let old_off_abs_vec = vabsq_f64(vdupq_n_f64(old_off));
    let rd_vec = vdupq_n_f64(rd);
    let best_dist_vec = vdupq_n_f64(best_dist);
    let zero_vec = vdupq_n_f64(0.0);

    let mut mask: u16 = 0;

    for chunk in 0..8 {
        let idx = chunk * 2;
        let lower = vld1q_f64(lower_vals.as_ptr().add(idx));
        let upper = vld1q_f64(upper_vals.as_ptr().add(idx));

        let below = vmaxq_f64(vsubq_f64(lower, query_vec), zero_vec);
        let above = vmaxq_f64(vsubq_f64(query_vec, upper), zero_vec);
        let interval = vaddq_f64(below, above);

        let delta = vsubq_f64(interval, old_off_abs_vec);
        let rd_far = vaddq_f64(rd_vec, delta);

        let cmp = vcleq_f64(rd_far, best_dist_vec);

        let weights = match chunk {
            0 => [1u64, 2u64],
            1 => [4u64, 8u64],
            2 => [16u64, 32u64],
            3 => [64u64, 128u64],
            4 => [256u64, 512u64],
            5 => [1024u64, 2048u64],
            6 => [4096u64, 8192u64],
            _ => [16384u64, 32768u64],
        };

        let mask_chunk = vaddvq_u64(vandq_u64(cmp, vld1q_u64(weights.as_ptr())));
        mask |= mask_chunk as u16;
    }

    mask
}

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
#[inline(always)]
#[allow(dead_code)]
unsafe fn simd_backtrack_block4_f32_neon_manhattan<
    A: Copy,
    D: DistanceMetricUnified<A, K, Output = f32>,
    const K: usize,
>(
    query_wide: f32,
    stems_ptr: NonNull<u8>,
    block_base_idx: usize,
    old_off: f32,
    rd: f32,
    best_dist: f32,
) -> u16 {
    use core::arch::aarch64::*;
    let _ = core::marker::PhantomData::<D>;

    let pivots_ptr = stems_ptr.as_ptr().add(block_base_idx * 4) as *const f32;
    let mut pivots = [0.0f32; 16];
    core::ptr::copy_nonoverlapping(pivots_ptr, pivots.as_mut_ptr(), 16);

    let mut lower_vals = [0.0f32; 16];
    let mut upper_vals = [0.0f32; 16];

    for i in 0..16 {
        let (lower_offset, upper_offset) = super::child_interval_bounds_block4(i);
        lower_vals[i] = if lower_offset == 255 {
            f32::NEG_INFINITY
        } else {
            pivots[lower_offset as usize]
        };
        upper_vals[i] = if upper_offset == 255 {
            f32::INFINITY
        } else {
            pivots[upper_offset as usize]
        };
    }

    let query_vec = vdupq_n_f32(query_wide);
    let old_off_abs_vec = vabsq_f32(vdupq_n_f32(old_off));
    let rd_vec = vdupq_n_f32(rd);
    let best_dist_vec = vdupq_n_f32(best_dist);
    let zero_vec = vdupq_n_f32(0.0);

    let mut mask: u16 = 0;

    for chunk in 0..4 {
        let idx = chunk * 4;
        let lower = vld1q_f32(lower_vals.as_ptr().add(idx));
        let upper = vld1q_f32(upper_vals.as_ptr().add(idx));

        let below = vmaxq_f32(vsubq_f32(lower, query_vec), zero_vec);
        let above = vmaxq_f32(vsubq_f32(query_vec, upper), zero_vec);
        let interval = vaddq_f32(below, above);

        let delta = vsubq_f32(interval, old_off_abs_vec);
        let rd_far = vaddq_f32(rd_vec, delta);

        let cmp = vcleq_f32(rd_far, best_dist_vec);

        let weights = match chunk {
            0 => [1u32, 2u32, 4u32, 8u32],
            1 => [16u32, 32u32, 64u32, 128u32],
            2 => [256u32, 512u32, 1024u32, 2048u32],
            _ => [4096u32, 8192u32, 16384u32, 32768u32],
        };

        let mask_chunk = vaddvq_u32(vandq_u32(cmp, vld1q_u32(weights.as_ptr())));
        mask |= mask_chunk as u16;
    }

    mask
}

// ====================================================================================
// Fixed-point implementations
// ====================================================================================

#[cfg(feature = "fixed")]
mod fixed_impls {
    use super::*;
    use fixed::{types::extra, FixedI32, FixedU16};

    type U0 = extra::U0;
    type U16 = extra::U16;
    type U8 = extra::U8;

    macro_rules! impl_backtrack_fixed {
        ($fixed_ty:ty) => {
            impl sealed::Sealed for $fixed_ty {}

            impl BacktrackBlock3 for $fixed_ty {
                #[inline(always)]
                fn backtrack_block3<A, D, const K: usize>(
                    query_wide: Self,
                    stems_ptr: NonNull<u8>,
                    block_base_idx: usize,
                    old_off: Self,
                    rd: Self,
                    best_dist: Self,
                ) -> u8
                where
                    A: Copy,
                    D: DistanceMetricUnified<A, K, Output = Self>
                        + DistanceMetricSimdBlock3<A, K, Self>,
                {
                    // TODO: Integer SIMD for fixed-point
                    autovec_backtrack_block3::<Self, A, D, K>(
                        query_wide,
                        stems_ptr,
                        block_base_idx,
                        old_off,
                        rd,
                        best_dist,
                    )
                }
            }

            impl BacktrackBlock4 for $fixed_ty {
                #[inline(always)]
                fn backtrack_block4<A, D, const K: usize>(
                    query_wide: Self,
                    stems_ptr: NonNull<u8>,
                    block_base_idx: usize,
                    old_off: Self,
                    rd: Self,
                    best_dist: Self,
                ) -> u16
                where
                    A: Copy,
                    D: DistanceMetricUnified<A, K, Output = Self>
                        + DistanceMetricSimdBlock4<A, K, Self>,
                {
                    // TODO: Integer SIMD for fixed-point
                    autovec_backtrack_block4::<Self, A, D, K>(
                        query_wide,
                        stems_ptr,
                        block_base_idx,
                        old_off,
                        rd,
                        best_dist,
                    )
                }
            }
        };
    }

    impl_backtrack_fixed!(FixedI32<U0>);
    impl_backtrack_fixed!(FixedI32<U16>);
    impl_backtrack_fixed!(FixedU16<U8>);
}

// ====================================================================================
// f16 implementations
// ====================================================================================

#[cfg(feature = "f16")]
mod f16_impl {
    use super::*;
    use half::f16;

    impl sealed::Sealed for f16 {}

    impl BacktrackBlock3 for f16 {
        #[inline(always)]
        fn backtrack_block3<A, D, const K: usize>(
            query_wide: Self,
            stems_ptr: NonNull<u8>,
            block_base_idx: usize,
            old_off: Self,
            rd: Self,
            best_dist: Self,
        ) -> u8
        where
            A: Copy,
            D: DistanceMetricUnified<A, K, Output = Self> + DistanceMetricSimdBlock3<A, K, Self>,
        {
            // TODO: f16 SIMD (possibly widen to f32)
            autovec_backtrack_block3::<Self, A, D, K>(
                query_wide,
                stems_ptr,
                block_base_idx,
                old_off,
                rd,
                best_dist,
            )
        }
    }

    impl BacktrackBlock4 for f16 {
        #[inline(always)]
        fn backtrack_block4<A, D, const K: usize>(
            query_wide: Self,
            stems_ptr: NonNull<u8>,
            block_base_idx: usize,
            old_off: Self,
            rd: Self,
            best_dist: Self,
        ) -> u16
        where
            A: Copy,
            D: DistanceMetricUnified<A, K, Output = Self> + DistanceMetricSimdBlock4<A, K, Self>,
        {
            // TODO: f16 SIMD (possibly widen to f32)
            autovec_backtrack_block4::<Self, A, D, K>(
                query_wide,
                stems_ptr,
                block_base_idx,
                old_off,
                rd,
                best_dist,
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stem_strategies::donnelly_2_blockmarker_simd::{
        child_interval_bounds_block3, child_interval_bounds_block4, interval_distance_1d,
    };
    use crate::traits_unified_2::{DistanceMetricUnified, SquaredEuclidean};

    #[cfg(all(feature = "simd", any(target_arch = "x86_64", target_arch = "aarch64")))]
    struct Lcg {
        state: u64,
    }

    #[cfg(all(feature = "simd", any(target_arch = "x86_64", target_arch = "aarch64")))]
    impl Lcg {
        fn new(seed: u64) -> Self {
            Self { state: seed }
        }

        fn next_u64(&mut self) -> u64 {
            self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
            self.state
        }

        fn next_f64(&mut self) -> f64 {
            let bits = self.next_u64() >> 11;
            let scale = (1u64 << 53) as f64;
            (bits as f64) / scale
        }

        fn next_f32(&mut self) -> f32 {
            let bits = (self.next_u64() >> 40) as u32;
            let scale = (1u32 << 24) as f32;
            (bits as f32) / scale
        }

        fn range_f64(&mut self, min: f64, max: f64) -> f64 {
            min + (max - min) * self.next_f64()
        }

        fn range_f32(&mut self, min: f32, max: f32) -> f32 {
            min + (max - min) * self.next_f32()
        }
    }

    fn build_block3_pivots_f64() -> [f64; 8] {
        let mut pivots = [0.0f64; 8];
        // Block3 BFS order: p0, p1, p2, p3, p4, p5, p6, padding
        // In-order: p3, p1, p4, p0, p5, p2, p6
        pivots[0] = 4.0; // p0
        pivots[1] = 2.0; // p1
        pivots[2] = 6.0; // p2
        pivots[3] = 1.0; // p3
        pivots[4] = 3.0; // p4
        pivots[5] = 5.0; // p5
        pivots[6] = 7.0; // p6
        pivots[7] = f64::MAX;
        pivots
    }

    #[cfg(all(feature = "simd", any(target_arch = "x86_64", target_arch = "aarch64")))]
    fn build_block3_pivots_from_sorted_f64(sorted: &[f64; 7]) -> [f64; 8] {
        let mut pivots = [0.0f64; 8];
        pivots[0] = sorted[3]; // p0
        pivots[1] = sorted[1]; // p1
        pivots[2] = sorted[5]; // p2
        pivots[3] = sorted[0]; // p3
        pivots[4] = sorted[2]; // p4
        pivots[5] = sorted[4]; // p5
        pivots[6] = sorted[6]; // p6
        pivots[7] = f64::MAX;
        pivots
    }

    fn build_block3_pivots_f32() -> [f32; 8] {
        let mut pivots = [0.0f32; 8];
        pivots[0] = 4.0;
        pivots[1] = 2.0;
        pivots[2] = 6.0;
        pivots[3] = 1.0;
        pivots[4] = 3.0;
        pivots[5] = 5.0;
        pivots[6] = 7.0;
        pivots[7] = f32::MAX;
        pivots
    }

    #[cfg(all(feature = "simd", any(target_arch = "x86_64", target_arch = "aarch64")))]
    fn build_block3_pivots_from_sorted_f32(sorted: &[f32; 7]) -> [f32; 8] {
        let mut pivots = [0.0f32; 8];
        pivots[0] = sorted[3];
        pivots[1] = sorted[1];
        pivots[2] = sorted[5];
        pivots[3] = sorted[0];
        pivots[4] = sorted[2];
        pivots[5] = sorted[4];
        pivots[6] = sorted[6];
        pivots[7] = f32::MAX;
        pivots
    }

    fn build_block4_pivots_f64() -> [f64; 16] {
        let mut pivots = [0.0f64; 16];
        for i in 0..15 {
            pivots[i] = (i + 1) as f64;
        }
        pivots[15] = f64::MAX;
        pivots
    }

    #[cfg(all(feature = "simd", any(target_arch = "x86_64", target_arch = "aarch64")))]
    fn build_block4_pivots_from_sorted_f64(sorted: &[f64; 15]) -> [f64; 16] {
        let mut pivots = [0.0f64; 16];
        pivots[7] = sorted[0];
        pivots[3] = sorted[1];
        pivots[8] = sorted[2];
        pivots[1] = sorted[3];
        pivots[9] = sorted[4];
        pivots[4] = sorted[5];
        pivots[10] = sorted[6];
        pivots[0] = sorted[7];
        pivots[11] = sorted[8];
        pivots[5] = sorted[9];
        pivots[12] = sorted[10];
        pivots[2] = sorted[11];
        pivots[13] = sorted[12];
        pivots[6] = sorted[13];
        pivots[14] = sorted[14];
        pivots[15] = f64::MAX;
        pivots
    }

    fn build_block4_pivots_f32() -> [f32; 16] {
        let mut pivots = [0.0f32; 16];
        for i in 0..15 {
            pivots[i] = (i + 1) as f32;
        }
        pivots[15] = f32::MAX;
        pivots
    }

    #[cfg(all(feature = "simd", any(target_arch = "x86_64", target_arch = "aarch64")))]
    fn build_block4_pivots_from_sorted_f32(sorted: &[f32; 15]) -> [f32; 16] {
        let mut pivots = [0.0f32; 16];
        pivots[7] = sorted[0];
        pivots[3] = sorted[1];
        pivots[8] = sorted[2];
        pivots[1] = sorted[3];
        pivots[9] = sorted[4];
        pivots[4] = sorted[5];
        pivots[10] = sorted[6];
        pivots[0] = sorted[7];
        pivots[11] = sorted[8];
        pivots[5] = sorted[9];
        pivots[12] = sorted[10];
        pivots[2] = sorted[11];
        pivots[13] = sorted[12];
        pivots[6] = sorted[13];
        pivots[14] = sorted[14];
        pivots[15] = f32::MAX;
        pivots
    }

    fn scalar_backtrack_block3_f64(
        query: f64,
        pivots: &[f64; 8],
        old_off: f64,
        rd: f64,
        best_dist: f64,
    ) -> u8 {
        let mut mask: u8 = 0;
        for sibling_idx in 0..8u8 {
            let (lower_offset, upper_offset) = child_interval_bounds_block3(sibling_idx as usize);

            let lower = if lower_offset == 255 {
                f64::MIN
            } else {
                pivots[lower_offset as usize]
            };

            let upper = if upper_offset == 255 {
                f64::MAX
            } else {
                pivots[upper_offset as usize]
            };

            let interval_dist = interval_distance_1d(query, lower, upper);
            let new_off =
                <SquaredEuclidean<f64> as DistanceMetricUnified<f64, 3>>::dist1(interval_dist, 0.0);
            let rd_far = rd - old_off + new_off;

            if rd_far <= best_dist {
                mask |= 1u8 << sibling_idx;
            }
        }
        mask
    }

    fn scalar_backtrack_block3_f32(
        query: f32,
        pivots: &[f32; 8],
        old_off: f32,
        rd: f32,
        best_dist: f32,
    ) -> u8 {
        let mut mask: u8 = 0;
        for sibling_idx in 0..8u8 {
            let (lower_offset, upper_offset) = child_interval_bounds_block3(sibling_idx as usize);

            let lower = if lower_offset == 255 {
                f32::MIN
            } else {
                pivots[lower_offset as usize]
            };

            let upper = if upper_offset == 255 {
                f32::MAX
            } else {
                pivots[upper_offset as usize]
            };

            let interval_dist = interval_distance_1d(query, lower, upper);
            let new_off =
                <SquaredEuclidean<f32> as DistanceMetricUnified<f32, 3>>::dist1(interval_dist, 0.0);
            let rd_far = rd - old_off + new_off;

            if rd_far <= best_dist {
                mask |= 1u8 << sibling_idx;
            }
        }
        mask
    }

    fn scalar_backtrack_block4_f64(
        query: f64,
        pivots: &[f64; 16],
        old_off: f64,
        rd: f64,
        best_dist: f64,
    ) -> u16 {
        let mut mask: u16 = 0;
        for sibling_idx in 0..16u8 {
            let (lower_offset, upper_offset) = child_interval_bounds_block4(sibling_idx as usize);

            let lower = if lower_offset == 255 {
                f64::MIN
            } else {
                pivots[lower_offset as usize]
            };

            let upper = if upper_offset == 255 {
                f64::MAX
            } else {
                pivots[upper_offset as usize]
            };

            let interval_dist = interval_distance_1d(query, lower, upper);
            let new_off =
                <SquaredEuclidean<f64> as DistanceMetricUnified<f64, 3>>::dist1(interval_dist, 0.0);
            let rd_far = rd - old_off + new_off;

            if rd_far <= best_dist {
                mask |= 1u16 << sibling_idx;
            }
        }
        mask
    }

    fn scalar_backtrack_block4_f32(
        query: f32,
        pivots: &[f32; 16],
        old_off: f32,
        rd: f32,
        best_dist: f32,
    ) -> u16 {
        let mut mask: u16 = 0;
        for sibling_idx in 0..16u8 {
            let (lower_offset, upper_offset) = child_interval_bounds_block4(sibling_idx as usize);

            let lower = if lower_offset == 255 {
                f32::MIN
            } else {
                pivots[lower_offset as usize]
            };

            let upper = if upper_offset == 255 {
                f32::MAX
            } else {
                pivots[upper_offset as usize]
            };

            let interval_dist = interval_distance_1d(query, lower, upper);
            let new_off =
                <SquaredEuclidean<f32> as DistanceMetricUnified<f32, 3>>::dist1(interval_dist, 0.0);
            let rd_far = rd - old_off + new_off;

            if rd_far <= best_dist {
                mask |= 1u16 << sibling_idx;
            }
        }
        mask
    }

    #[test]
    fn test_backtrack_block3_f64_basic() {
        // Create a simple test case with known pivots
        let mut pivots = [0.0f64; 8];
        // Block3 BST order: p3, p1, p4, p0, p5, p2, p6, (padding)
        // In-order: p0=1.0, p1=2.0, p2=3.0, p3=4.0, p4=5.0, p5=6.0, p6=7.0
        pivots[0] = 4.0; // p3
        pivots[1] = 2.0; // p1
        pivots[2] = 6.0; // p4
        pivots[3] = 1.0; // p0
        pivots[4] = 5.0; // p2
        pivots[5] = 3.0; // p5
        pivots[6] = 7.0; // p6
        pivots[7] = f64::MAX; // padding

        let stems_ptr = NonNull::new(pivots.as_mut_ptr() as *mut u8).unwrap();

        // Query at 4.5, which should be close to pivots around 4-5
        let mask = f64::backtrack_block3::<f64, SquaredEuclidean<f64>, 3>(
            4.5,       // query
            stems_ptr, // stems
            0,         // block_base_idx
            0.0,       // old_off
            0.0,       // rd
            10.0,      // best_dist (generous)
        );

        // With best_dist=10.0 and query=4.5, most siblings should be reachable
        assert!(mask != 0, "Should have some reachable siblings");
    }

    #[test]
    fn test_backtrack_block3_f32_basic() {
        let mut pivots = [0.0f32; 8];
        pivots[0] = 4.0;
        pivots[1] = 2.0;
        pivots[2] = 6.0;
        pivots[3] = 1.0;
        pivots[4] = 5.0;
        pivots[5] = 3.0;
        pivots[6] = 7.0;
        pivots[7] = f32::MAX;

        let stems_ptr = NonNull::new(pivots.as_mut_ptr() as *mut u8).unwrap();

        let mask = f32::backtrack_block3::<f32, SquaredEuclidean<f32>, 3>(
            4.5, stems_ptr, 0, 0.0, 0.0, 10.0,
        );

        assert!(mask != 0, "Should have some reachable siblings");
    }

    #[test]
    fn test_block3_backtrack_interval_correctness() {
        let mut pivots_f64 = build_block3_pivots_f64();
        let stems_ptr_f64 = NonNull::new(pivots_f64.as_mut_ptr() as *mut u8).unwrap();

        let cases_f64 = [
            (4.5, 0.0, 0.0, 4.0),
            (1.2, 0.5, 1.0, 2.5),
            (6.9, 0.0, 0.0, 1.0),
        ];

        for (query, old_off, rd, best_dist) in cases_f64 {
            let expected = scalar_backtrack_block3_f64(query, &pivots_f64, old_off, rd, best_dist);
            let actual = autovec_backtrack_block3::<f64, f64, SquaredEuclidean<f64>, 3>(
                query,
                stems_ptr_f64,
                0,
                old_off,
                rd,
                best_dist,
            );
            assert_eq!(
                actual, expected,
                "query={query}, old_off={old_off}, rd={rd}, best_dist={best_dist}"
            );
        }

        let mut pivots_f32 = build_block3_pivots_f32();
        let stems_ptr_f32 = NonNull::new(pivots_f32.as_mut_ptr() as *mut u8).unwrap();

        let cases_f32 = [
            (4.5f32, 0.0f32, 0.0f32, 4.0f32),
            (1.2f32, 0.5f32, 1.0f32, 2.5f32),
            (6.9f32, 0.0f32, 0.0f32, 1.0f32),
        ];

        for (query, old_off, rd, best_dist) in cases_f32 {
            let expected = scalar_backtrack_block3_f32(query, &pivots_f32, old_off, rd, best_dist);
            let actual = autovec_backtrack_block3::<f32, f32, SquaredEuclidean<f32>, 3>(
                query,
                stems_ptr_f32,
                0,
                old_off,
                rd,
                best_dist,
            );
            assert_eq!(
                actual, expected,
                "query={query}, old_off={old_off}, rd={rd}, best_dist={best_dist}"
            );
        }
    }

    #[test]
    fn test_backtrack_block4_f64_correctness() {
        let mut pivots = build_block4_pivots_f64();
        let stems_ptr = NonNull::new(pivots.as_mut_ptr() as *mut u8).unwrap();

        let cases = [
            (4.5, 0.0, 0.0, 4.0),
            (9.2, 1.0, 2.0, 9.0),
            (0.25, 0.0, 0.0, 0.5),
        ];

        for (query, old_off, rd, best_dist) in cases {
            let expected = scalar_backtrack_block4_f64(query, &pivots, old_off, rd, best_dist);
            let actual = f64::backtrack_block4::<f64, SquaredEuclidean<f64>, 3>(
                query, stems_ptr, 0, old_off, rd, best_dist,
            );
            assert_eq!(
                actual, expected,
                "query={query}, old_off={old_off}, rd={rd}, best_dist={best_dist}"
            );
        }
    }

    #[test]
    fn test_backtrack_block4_f32_correctness() {
        let mut pivots = build_block4_pivots_f32();
        let stems_ptr = NonNull::new(pivots.as_mut_ptr() as *mut u8).unwrap();

        let cases = [
            (4.5f32, 0.0f32, 0.0f32, 4.0f32),
            (9.2f32, 1.0f32, 2.0f32, 9.0f32),
            (0.25f32, 0.0f32, 0.0f32, 0.5f32),
        ];

        for (query, old_off, rd, best_dist) in cases {
            let expected = scalar_backtrack_block4_f32(query, &pivots, old_off, rd, best_dist);
            let actual = f32::backtrack_block4::<f32, SquaredEuclidean<f32>, 3>(
                query, stems_ptr, 0, old_off, rd, best_dist,
            );
            assert_eq!(
                actual, expected,
                "query={query}, old_off={old_off}, rd={rd}, best_dist={best_dist}"
            );
        }
    }

    #[cfg(all(feature = "simd", any(target_arch = "x86_64", target_arch = "aarch64")))]
    #[test]
    fn test_block4_f64_simd_vs_autovec() {
        let mut pivots = build_block4_pivots_f64();
        let stems_ptr = NonNull::new(pivots.as_mut_ptr() as *mut u8).unwrap();

        let cases = [
            (4.5, 0.0, 0.0, 4.0),
            (9.2, 1.0, 2.0, 9.0),
            (0.25, 0.0, 0.0, 0.5),
        ];

        for (query, old_off, rd, best_dist) in cases {
            let simd_mask = f64::backtrack_block4::<f64, SquaredEuclidean<f64>, 3>(
                query, stems_ptr, 0, old_off, rd, best_dist,
            );
            let autovec_mask = autovec_backtrack_block4::<f64, f64, SquaredEuclidean<f64>, 3>(
                query, stems_ptr, 0, old_off, rd, best_dist,
            );
            assert_eq!(
                simd_mask, autovec_mask,
                "query={query}, old_off={old_off}, rd={rd}, best_dist={best_dist}"
            );
        }
    }

    #[cfg(all(feature = "simd", any(target_arch = "x86_64", target_arch = "aarch64")))]
    #[test]
    fn test_block4_f32_simd_vs_autovec() {
        let mut pivots = build_block4_pivots_f32();
        let stems_ptr = NonNull::new(pivots.as_mut_ptr() as *mut u8).unwrap();

        let cases = [
            (4.5f32, 0.0f32, 0.0f32, 4.0f32),
            (9.2f32, 1.0f32, 2.0f32, 9.0f32),
            (0.25f32, 0.0f32, 0.0f32, 0.5f32),
        ];

        for (query, old_off, rd, best_dist) in cases {
            let simd_mask = f32::backtrack_block4::<f32, SquaredEuclidean<f32>, 3>(
                query, stems_ptr, 0, old_off, rd, best_dist,
            );
            let autovec_mask = autovec_backtrack_block4::<f32, f32, SquaredEuclidean<f32>, 3>(
                query, stems_ptr, 0, old_off, rd, best_dist,
            );
            assert_eq!(
                simd_mask, autovec_mask,
                "query={query}, old_off={old_off}, rd={rd}, best_dist={best_dist}"
            );
        }
    }

    #[cfg(all(feature = "simd", any(target_arch = "x86_64", target_arch = "aarch64")))]
    #[test]
    fn test_block3_f64_simd_vs_autovec() {
        let mut pivots = build_block3_pivots_f64();
        let stems_ptr = NonNull::new(pivots.as_mut_ptr() as *mut u8).unwrap();

        let cases = [
            (4.5, 0.0, 0.0, 4.0),
            (1.2, 0.5, 1.0, 2.5),
            (6.9, 0.0, 0.0, 1.0),
        ];

        for (query, old_off, rd, best_dist) in cases {
            let simd_mask = f64::backtrack_block3::<f64, SquaredEuclidean<f64>, 3>(
                query, stems_ptr, 0, old_off, rd, best_dist,
            );
            let autovec_mask = autovec_backtrack_block3::<f64, f64, SquaredEuclidean<f64>, 3>(
                query, stems_ptr, 0, old_off, rd, best_dist,
            );
            assert_eq!(
                simd_mask, autovec_mask,
                "query={query}, old_off={old_off}, rd={rd}, best_dist={best_dist}"
            );
        }
    }

    #[cfg(all(feature = "simd", any(target_arch = "x86_64", target_arch = "aarch64")))]
    #[test]
    fn test_block3_f32_simd_vs_autovec() {
        let mut pivots = build_block3_pivots_f32();
        let stems_ptr = NonNull::new(pivots.as_mut_ptr() as *mut u8).unwrap();

        let cases = [
            (4.5f32, 0.0f32, 0.0f32, 4.0f32),
            (1.2f32, 0.5f32, 1.0f32, 2.5f32),
            (6.9f32, 0.0f32, 0.0f32, 1.0f32),
        ];

        for (query, old_off, rd, best_dist) in cases {
            let simd_mask = f32::backtrack_block3::<f32, SquaredEuclidean<f32>, 3>(
                query, stems_ptr, 0, old_off, rd, best_dist,
            );
            let autovec_mask = autovec_backtrack_block3::<f32, f32, SquaredEuclidean<f32>, 3>(
                query, stems_ptr, 0, old_off, rd, best_dist,
            );
            assert_eq!(
                simd_mask, autovec_mask,
                "query={query}, old_off={old_off}, rd={rd}, best_dist={best_dist}"
            );
        }
    }

    #[cfg(all(feature = "simd", any(target_arch = "x86_64", target_arch = "aarch64")))]
    #[test]
    fn test_block3_f64_manhattan_simd_vs_autovec() {
        let mut pivots = build_block3_pivots_f64();
        let stems_ptr = NonNull::new(pivots.as_mut_ptr() as *mut u8).unwrap();

        let cases = [
            (4.5, 0.0, 0.0, 4.0),
            (1.2, 0.5, 1.0, 2.5),
            (6.9, 0.0, 0.0, 1.0),
            (3.7, -1.25, 2.0, 6.0),
        ];

        for (query, old_off, rd, best_dist) in cases {
            let simd_mask = f64::backtrack_block3::<f64, Manhattan<f64>, 3>(
                query, stems_ptr, 0, old_off, rd, best_dist,
            );
            let autovec_mask = autovec_backtrack_block3::<f64, f64, Manhattan<f64>, 3>(
                query, stems_ptr, 0, old_off, rd, best_dist,
            );
            assert_eq!(
                simd_mask, autovec_mask,
                "query={query}, old_off={old_off}, rd={rd}, best_dist={best_dist}"
            );
        }
    }

    #[cfg(all(feature = "simd", any(target_arch = "x86_64", target_arch = "aarch64")))]
    #[test]
    fn test_block3_f32_manhattan_simd_vs_autovec() {
        let mut pivots = build_block3_pivots_f32();
        let stems_ptr = NonNull::new(pivots.as_mut_ptr() as *mut u8).unwrap();

        let cases = [
            (4.5f32, 0.0f32, 0.0f32, 4.0f32),
            (1.2f32, 0.5f32, 1.0f32, 2.5f32),
            (6.9f32, 0.0f32, 0.0f32, 1.0f32),
            (3.7f32, -1.25f32, 2.0f32, 6.0f32),
        ];

        for (query, old_off, rd, best_dist) in cases {
            let simd_mask = f32::backtrack_block3::<f32, Manhattan<f32>, 3>(
                query, stems_ptr, 0, old_off, rd, best_dist,
            );
            let autovec_mask = autovec_backtrack_block3::<f32, f32, Manhattan<f32>, 3>(
                query, stems_ptr, 0, old_off, rd, best_dist,
            );
            assert_eq!(
                simd_mask, autovec_mask,
                "query={query}, old_off={old_off}, rd={rd}, best_dist={best_dist}"
            );
        }
    }

    #[cfg(all(feature = "simd", any(target_arch = "x86_64", target_arch = "aarch64")))]
    #[test]
    fn test_block4_f64_manhattan_simd_vs_autovec() {
        let mut pivots = build_block4_pivots_f64();
        let stems_ptr = NonNull::new(pivots.as_mut_ptr() as *mut u8).unwrap();

        let cases = [
            (4.5, 0.0, 0.0, 4.0),
            (9.2, 1.0, 2.0, 9.0),
            (0.25, 0.0, 0.0, 0.5),
            (6.7, -0.75, 1.5, 4.0),
        ];

        for (query, old_off, rd, best_dist) in cases {
            let simd_mask = f64::backtrack_block4::<f64, Manhattan<f64>, 3>(
                query, stems_ptr, 0, old_off, rd, best_dist,
            );
            let autovec_mask = autovec_backtrack_block4::<f64, f64, Manhattan<f64>, 3>(
                query, stems_ptr, 0, old_off, rd, best_dist,
            );
            assert_eq!(
                simd_mask, autovec_mask,
                "query={query}, old_off={old_off}, rd={rd}, best_dist={best_dist}"
            );
        }
    }

    #[cfg(all(feature = "simd", any(target_arch = "x86_64", target_arch = "aarch64")))]
    #[test]
    fn test_block4_f32_manhattan_simd_vs_autovec() {
        let mut pivots = build_block4_pivots_f32();
        let stems_ptr = NonNull::new(pivots.as_mut_ptr() as *mut u8).unwrap();

        let cases = [
            (4.5f32, 0.0f32, 0.0f32, 4.0f32),
            (9.2f32, 1.0f32, 2.0f32, 9.0f32),
            (0.25f32, 0.0f32, 0.0f32, 0.5f32),
            (6.7f32, -0.75f32, 1.5f32, 4.0f32),
        ];

        for (query, old_off, rd, best_dist) in cases {
            let simd_mask = f32::backtrack_block4::<f32, Manhattan<f32>, 3>(
                query, stems_ptr, 0, old_off, rd, best_dist,
            );
            let autovec_mask = autovec_backtrack_block4::<f32, f32, Manhattan<f32>, 3>(
                query, stems_ptr, 0, old_off, rd, best_dist,
            );
            assert_eq!(
                simd_mask, autovec_mask,
                "query={query}, old_off={old_off}, rd={rd}, best_dist={best_dist}"
            );
        }
    }

    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    #[test]
    fn test_block3_neon_correctness() {
        let mut pivots_f64 = build_block3_pivots_f64();
        let stems_ptr_f64 = NonNull::new(pivots_f64.as_mut_ptr() as *mut u8).unwrap();

        let cases_f64 = [
            (4.5, 0.0, 0.0, 4.0),
            (1.2, 0.5, 1.0, 2.5),
            (6.9, 0.0, 0.0, 1.0),
        ];

        for (query, old_off, rd, best_dist) in cases_f64 {
            let neon_mask = f64::backtrack_block3::<f64, SquaredEuclidean<f64>, 3>(
                query,
                stems_ptr_f64,
                0,
                old_off,
                rd,
                best_dist,
            );
            let autovec_mask = autovec_backtrack_block3::<f64, f64, SquaredEuclidean<f64>, 3>(
                query,
                stems_ptr_f64,
                0,
                old_off,
                rd,
                best_dist,
            );
            assert_eq!(
                neon_mask, autovec_mask,
                "query={query}, old_off={old_off}, rd={rd}, best_dist={best_dist}"
            );
        }

        let mut pivots_f32 = build_block3_pivots_f32();
        let stems_ptr_f32 = NonNull::new(pivots_f32.as_mut_ptr() as *mut u8).unwrap();

        let cases_f32 = [
            (4.5f32, 0.0f32, 0.0f32, 4.0f32),
            (1.2f32, 0.5f32, 1.0f32, 2.5f32),
            (6.9f32, 0.0f32, 0.0f32, 1.0f32),
        ];

        for (query, old_off, rd, best_dist) in cases_f32 {
            let neon_mask = f32::backtrack_block3::<f32, SquaredEuclidean<f32>, 3>(
                query,
                stems_ptr_f32,
                0,
                old_off,
                rd,
                best_dist,
            );
            let autovec_mask = autovec_backtrack_block3::<f32, f32, SquaredEuclidean<f32>, 3>(
                query,
                stems_ptr_f32,
                0,
                old_off,
                rd,
                best_dist,
            );
            assert_eq!(
                neon_mask, autovec_mask,
                "query={query}, old_off={old_off}, rd={rd}, best_dist={best_dist}"
            );
        }
    }

    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    #[test]
    fn test_block4_neon_correctness() {
        let mut pivots_f64 = build_block4_pivots_f64();
        let stems_ptr_f64 = NonNull::new(pivots_f64.as_mut_ptr() as *mut u8).unwrap();

        let cases_f64 = [
            (4.5, 0.0, 0.0, 4.0),
            (9.2, 1.0, 2.0, 9.0),
            (0.25, 0.0, 0.0, 0.5),
        ];

        for (query, old_off, rd, best_dist) in cases_f64 {
            let neon_mask = f64::backtrack_block4::<f64, SquaredEuclidean<f64>, 3>(
                query,
                stems_ptr_f64,
                0,
                old_off,
                rd,
                best_dist,
            );
            let autovec_mask = autovec_backtrack_block4::<f64, f64, SquaredEuclidean<f64>, 3>(
                query,
                stems_ptr_f64,
                0,
                old_off,
                rd,
                best_dist,
            );
            assert_eq!(
                neon_mask, autovec_mask,
                "query={query}, old_off={old_off}, rd={rd}, best_dist={best_dist}"
            );
        }

        let mut pivots_f32 = build_block4_pivots_f32();
        let stems_ptr_f32 = NonNull::new(pivots_f32.as_mut_ptr() as *mut u8).unwrap();

        let cases_f32 = [
            (4.5f32, 0.0f32, 0.0f32, 4.0f32),
            (9.2f32, 1.0f32, 2.0f32, 9.0f32),
            (0.25f32, 0.0f32, 0.0f32, 0.5f32),
        ];

        for (query, old_off, rd, best_dist) in cases_f32 {
            let neon_mask = f32::backtrack_block4::<f32, SquaredEuclidean<f32>, 3>(
                query,
                stems_ptr_f32,
                0,
                old_off,
                rd,
                best_dist,
            );
            let autovec_mask = autovec_backtrack_block4::<f32, f32, SquaredEuclidean<f32>, 3>(
                query,
                stems_ptr_f32,
                0,
                old_off,
                rd,
                best_dist,
            );
            assert_eq!(
                neon_mask, autovec_mask,
                "query={query}, old_off={old_off}, rd={rd}, best_dist={best_dist}"
            );
        }
    }

    #[cfg(all(feature = "simd", any(target_arch = "x86_64", target_arch = "aarch64")))]
    #[test]
    fn test_block3_backtrack_property_based() {
        const ITERATIONS: usize = 512;
        let mut rng = Lcg::new(0x5eeda11u64);

        for _ in 0..ITERATIONS {
            let mut sorted = [0.0f64; 7];
            let mut current = rng.range_f64(-5.0, 5.0);
            for val in sorted.iter_mut() {
                current += rng.range_f64(0.1, 3.0);
                *val = current;
            }
            let mut pivots = build_block3_pivots_from_sorted_f64(&sorted);
            let stems_ptr = NonNull::new(pivots.as_mut_ptr() as *mut u8).unwrap();

            let query = rng.range_f64(sorted[0] - 2.0, sorted[6] + 2.0);
            let old_off = rng.range_f64(0.0, 5.0);
            let rd = rng.range_f64(0.0, 5.0);
            let best_dist = rng.range_f64(0.0, 10.0);

            let simd_mask = f64::backtrack_block3::<f64, SquaredEuclidean<f64>, 3>(
                query, stems_ptr, 0, old_off, rd, best_dist,
            );
            let autovec_mask = autovec_backtrack_block3::<f64, f64, SquaredEuclidean<f64>, 3>(
                query, stems_ptr, 0, old_off, rd, best_dist,
            );

            assert_eq!(
                simd_mask, autovec_mask,
                "f64 query={query}, old_off={old_off}, rd={rd}, best_dist={best_dist}"
            );
        }

        for _ in 0..ITERATIONS {
            let mut sorted = [0.0f32; 7];
            let mut current = rng.range_f32(-5.0, 5.0);
            for val in sorted.iter_mut() {
                current += rng.range_f32(0.1, 3.0);
                *val = current;
            }
            let mut pivots = build_block3_pivots_from_sorted_f32(&sorted);
            let stems_ptr = NonNull::new(pivots.as_mut_ptr() as *mut u8).unwrap();

            let query = rng.range_f32(sorted[0] - 2.0, sorted[6] + 2.0);
            let old_off = rng.range_f32(0.0, 5.0);
            let rd = rng.range_f32(0.0, 5.0);
            let best_dist = rng.range_f32(0.0, 10.0);

            let simd_mask = f32::backtrack_block3::<f32, SquaredEuclidean<f32>, 3>(
                query, stems_ptr, 0, old_off, rd, best_dist,
            );
            let autovec_mask = autovec_backtrack_block3::<f32, f32, SquaredEuclidean<f32>, 3>(
                query, stems_ptr, 0, old_off, rd, best_dist,
            );

            assert_eq!(
                simd_mask, autovec_mask,
                "f32 query={query}, old_off={old_off}, rd={rd}, best_dist={best_dist}"
            );
        }
    }

    #[cfg(all(feature = "simd", any(target_arch = "x86_64", target_arch = "aarch64")))]
    #[test]
    fn test_block4_backtrack_property_based() {
        const ITERATIONS: usize = 512;
        let mut rng = Lcg::new(0xfeedfaceu64);

        for _ in 0..ITERATIONS {
            let mut sorted = [0.0f64; 15];
            let mut current = rng.range_f64(-5.0, 5.0);
            for val in sorted.iter_mut() {
                current += rng.range_f64(0.1, 3.0);
                *val = current;
            }
            let mut pivots = build_block4_pivots_from_sorted_f64(&sorted);
            let stems_ptr = NonNull::new(pivots.as_mut_ptr() as *mut u8).unwrap();

            let query = rng.range_f64(sorted[0] - 2.0, sorted[14] + 2.0);
            let old_off = rng.range_f64(0.0, 5.0);
            let rd = rng.range_f64(0.0, 5.0);
            let best_dist = rng.range_f64(0.0, 10.0);

            let simd_mask = f64::backtrack_block4::<f64, SquaredEuclidean<f64>, 3>(
                query, stems_ptr, 0, old_off, rd, best_dist,
            );
            let autovec_mask = autovec_backtrack_block4::<f64, f64, SquaredEuclidean<f64>, 3>(
                query, stems_ptr, 0, old_off, rd, best_dist,
            );

            assert_eq!(
                simd_mask, autovec_mask,
                "f64 query={query}, old_off={old_off}, rd={rd}, best_dist={best_dist}"
            );
        }

        for _ in 0..ITERATIONS {
            let mut sorted = [0.0f32; 15];
            let mut current = rng.range_f32(-5.0, 5.0);
            for val in sorted.iter_mut() {
                current += rng.range_f32(0.1, 3.0);
                *val = current;
            }
            let mut pivots = build_block4_pivots_from_sorted_f32(&sorted);
            let stems_ptr = NonNull::new(pivots.as_mut_ptr() as *mut u8).unwrap();

            let query = rng.range_f32(sorted[0] - 2.0, sorted[14] + 2.0);
            let old_off = rng.range_f32(0.0, 5.0);
            let rd = rng.range_f32(0.0, 5.0);
            let best_dist = rng.range_f32(0.0, 10.0);

            let simd_mask = f32::backtrack_block4::<f32, SquaredEuclidean<f32>, 3>(
                query, stems_ptr, 0, old_off, rd, best_dist,
            );
            let autovec_mask = autovec_backtrack_block4::<f32, f32, SquaredEuclidean<f32>, 3>(
                query, stems_ptr, 0, old_off, rd, best_dist,
            );

            assert_eq!(
                simd_mask, autovec_mask,
                "f32 query={query}, old_off={old_off}, rd={rd}, best_dist={best_dist}"
            );
        }
    }
}
