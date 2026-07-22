//! Benchmark-only embedded leaf descriptor experiment.

use std::ptr::NonNull;

use aligned_vec::{AVec, CACHELINE_ALIGN};

use crate::dist::DistanceMetric;
use crate::kd_tree::{ConstructionError, KdTree};
use crate::leaf_strategy::VecOfArenas;
use crate::stem_strategy::donnelly::simd_full::compare_block3;
use crate::stem_strategy::{DonnellySimdDescentLeafEmbedded3, DonnellyUnrolledLeafEmbedded3};
use crate::{LeafStrategy, StemStrategy};

const DESCRIPTOR_OFFSET_BITS: u32 = 48;
const DESCRIPTOR_OFFSET_MASK: u64 = (1u64 << DESCRIPTOR_OFFSET_BITS) - 1;
const FINAL_PIVOT_LEVELS: i32 = 2;

type ConstructionTree<const K: usize, const B: usize> =
    KdTree<f64, u32, DonnellyUnrolledLeafEmbedded3, VecOfArenas<f64, u32, K, B>, K, B>;
type SimdConstructionTree<const K: usize, const B: usize> =
    KdTree<f64, u32, DonnellySimdDescentLeafEmbedded3, VecOfArenas<f64, u32, K, B>, K, B>;

/// Construction failure for the benchmark-only 48-bit-offset/16-bit-length layout.
#[doc(hidden)]
#[derive(Debug)]
pub enum EmbeddedLeafDescriptorError {
    /// The ordinary kd-tree construction failed.
    Construction(ConstructionError),
    /// A byte offset did not fit in the descriptor's low 48 bits.
    ByteOffsetOverflow { leaf_idx: usize, byte_offset: usize },
    /// A leaf length did not fit in the descriptor's high 16 bits.
    LeafLengthOverflow { leaf_idx: usize, leaf_len: usize },
}

impl From<ConstructionError> for EmbeddedLeafDescriptorError {
    fn from(value: ConstructionError) -> Self {
        Self::Construction(value)
    }
}

/// An aligned integer-word Donnelly tree used only by the descriptor benchmark.
///
/// Pivot words contain `f64::to_bits()` values. The last level of every final
/// block contains a packed 48-bit arena byte offset and 16-bit leaf length.
#[doc(hidden)]
pub struct EmbeddedLeafDescriptorF64Tree<const K: usize, const B: usize> {
    stem_words: AVec<u64>,
    leaves: VecOfArenas<f64, u32, K, B>,
    max_stem_level: i32,
    leaf_count: usize,
}

/// Descriptor tree using full SIMD-descent blocks and a shallow terminal block.
#[doc(hidden)]
pub struct EmbeddedLeafDescriptorSimdF64Tree<const K: usize, const B: usize> {
    stem_words: AVec<u64>,
    leaves: VecOfArenas<f64, u32, K, B>,
    max_stem_level: i32,
    leaf_count: usize,
}

impl<const K: usize, const B: usize> EmbeddedLeafDescriptorF64Tree<K, B> {
    /// Builds the benchmark tree while retaining the ordinary leaf extent table.
    pub fn new_from_slice(source: &[[f64; K]]) -> Result<Self, EmbeddedLeafDescriptorError> {
        let tree: ConstructionTree<K, B> = KdTree::new_from_slice(source)?;
        let KdTree {
            stems,
            leaves,
            max_stem_level,
            ..
        } = tree;

        let leaf_count = <VecOfArenas<f64, u32, K, B> as LeafStrategy<
            f64,
            u32,
            DonnellyUnrolledLeafEmbedded3,
            K,
            B,
        >>::leaf_count(&leaves);

        let mut stem_words = AVec::new(CACHELINE_ALIGN);
        stem_words.resize(stems.len(), 0);
        for (word, pivot) in stem_words.iter_mut().zip(stems.iter()) {
            *word = pivot.to_bits();
        }

        let pivot_depth = (max_stem_level + 1) as usize;
        debug_assert_eq!((pivot_depth + 1) % 3, 0);

        for leaf_idx in 0..leaf_count {
            let (byte_offset, leaf_len) = leaves.leaf_extent_for_embedded_descriptor(leaf_idx);
            let descriptor = pack_descriptor(leaf_idx, byte_offset, leaf_len)?;
            let terminal_stem_idx = terminal_stem_idx::<K>(leaf_idx, pivot_depth);
            debug_assert!(terminal_stem_idx < stem_words.len());
            unsafe {
                *stem_words.get_unchecked_mut(terminal_stem_idx) = descriptor;
            }
        }

        Ok(Self {
            stem_words,
            leaves,
            max_stem_level,
            leaf_count,
        })
    }

    /// Approximate nearest-one using the existing extent indirection.
    #[inline(always)]
    pub fn approx_nearest_one_via_extents<D>(&self, query: &[f64; K]) -> (f64, u32)
    where
        D: DistanceMetric<f64, Output = f64>,
    {
        let (leaf_idx, _) = self.descend(query);
        let arena = <VecOfArenas<f64, u32, K, B> as LeafStrategy<
            f64,
            u32,
            DonnellyUnrolledLeafEmbedded3,
            K,
            B,
        >>::leaf_arena(&self.leaves, leaf_idx);
        process_arena::<D, K>(&arena, query)
    }

    /// Approximate nearest-one using the descriptor selected from the final block.
    #[inline(always)]
    pub fn approx_nearest_one_embedded<D>(&self, query: &[f64; K]) -> (f64, u32)
    where
        D: DistanceMetric<f64, Output = f64>,
    {
        let (_, descriptor_idx) = self.descend(query);
        let descriptor = unsafe { *self.stem_words.get_unchecked(descriptor_idx) };
        let (byte_offset, leaf_len) = unpack_descriptor(descriptor);
        let arena = self
            .leaves
            .leaf_arena_from_embedded_descriptor(byte_offset, leaf_len);
        process_arena::<D, K>(&arena, query)
    }

    /// Number of logical leaves, exposed for benchmark validation.
    #[inline]
    pub fn leaf_count(&self) -> usize {
        self.leaf_count
    }

    /// Number of packed stem/descriptor words, exposed for benchmark diagnostics.
    #[inline]
    pub fn stem_word_count(&self) -> usize {
        self.stem_words.len()
    }

    #[inline(always)]
    fn descend(&self, query: &[f64; K]) -> (usize, usize) {
        let stems_ptr = NonNull::new(self.stem_words.as_ptr() as *mut u8).unwrap();
        let mut strat = DonnellyUnrolledLeafEmbedded3::new(stems_ptr);
        let pivot_depth = self.max_stem_level + 1;

        debug_assert_eq!((pivot_depth + 1) % 3, 0);

        while strat.level() + FINAL_PIVOT_LEVELS < pivot_depth {
            self.traverse_head(&mut strat, query);
            self.traverse_head(&mut strat, query);
            self.traverse_tail(&mut strat, query);
        }

        self.traverse_head(&mut strat, query);
        self.traverse_head(&mut strat, query);

        let leaf_idx = strat.leaf_idx();
        debug_assert!(leaf_idx < self.leaf_count);
        (leaf_idx, strat.stem_idx())
    }

    #[inline(always)]
    fn traverse_head(&self, strat: &mut DonnellyUnrolledLeafEmbedded3, query: &[f64; K]) {
        let pivot = f64::from_bits(unsafe { *self.stem_words.get_unchecked(strat.stem_idx()) });
        let query_value = unsafe { *query.get_unchecked(strat.dim::<K>()) };
        strat.traverse_head::<f64, K>(query_value >= pivot);
    }

    #[inline(always)]
    fn traverse_tail(&self, strat: &mut DonnellyUnrolledLeafEmbedded3, query: &[f64; K]) {
        let pivot = f64::from_bits(unsafe { *self.stem_words.get_unchecked(strat.stem_idx()) });
        let query_value = unsafe { *query.get_unchecked(strat.dim::<K>()) };
        strat.traverse_tail::<f64, K>(query_value >= pivot);
    }
}

impl<const K: usize, const B: usize> EmbeddedLeafDescriptorSimdF64Tree<K, B> {
    /// Builds the hybrid tree with block-dimension pivot construction.
    pub fn new_from_slice(source: &[[f64; K]]) -> Result<Self, EmbeddedLeafDescriptorError> {
        let tree: SimdConstructionTree<K, B> = KdTree::new_from_slice(source)?;
        let KdTree {
            stems,
            leaves,
            max_stem_level,
            ..
        } = tree;

        let leaf_count = <VecOfArenas<f64, u32, K, B> as LeafStrategy<
            f64,
            u32,
            DonnellySimdDescentLeafEmbedded3,
            K,
            B,
        >>::leaf_count(&leaves);

        let mut stem_words = AVec::new(CACHELINE_ALIGN);
        stem_words.resize(stems.len(), 0);
        for (word, pivot) in stem_words.iter_mut().zip(stems.iter()) {
            *word = pivot.to_bits();
        }

        let pivot_depth = (max_stem_level + 1) as usize;
        debug_assert_eq!((pivot_depth + 1) % 3, 0);

        for leaf_idx in 0..leaf_count {
            let (byte_offset, leaf_len) = leaves.leaf_extent_for_embedded_descriptor(leaf_idx);
            let descriptor = pack_descriptor(leaf_idx, byte_offset, leaf_len)?;
            let terminal_stem_idx = terminal_simd_stem_idx::<K>(leaf_idx, pivot_depth);
            debug_assert!(terminal_stem_idx < stem_words.len());
            unsafe {
                *stem_words.get_unchecked_mut(terminal_stem_idx) = descriptor;
            }
        }

        Ok(Self {
            stem_words,
            leaves,
            max_stem_level,
            leaf_count,
        })
    }

    /// Hybrid descent with a shallow SIMD selector and embedded descriptor.
    #[inline(always)]
    pub fn approx_nearest_one_embedded_shallow_simd<D>(&self, query: &[f64; K]) -> (f64, u32)
    where
        D: DistanceMetric<f64, Output = f64>,
    {
        let (strat, terminal_base_idx, query_value) = self.descend_complete_blocks(query);
        let terminal_rank =
            compare_terminal_block2_f64(&self.stem_words, terminal_base_idx, query_value);
        debug_assert!(strat.leaf_idx_with_terminal_rank(terminal_rank) < self.leaf_count);
        let descriptor = unsafe {
            *self
                .stem_words
                .get_unchecked(terminal_base_idx + 3 + usize::from(terminal_rank))
        };
        let (byte_offset, leaf_len) = unpack_descriptor(descriptor);
        let arena = self
            .leaves
            .leaf_arena_from_embedded_descriptor(byte_offset, leaf_len);
        process_arena::<D, K>(&arena, query)
    }

    /// Hybrid descent using two serial comparisons in the terminal block.
    #[inline(always)]
    pub fn approx_nearest_one_embedded_terminal_scalar<D>(&self, query: &[f64; K]) -> (f64, u32)
    where
        D: DistanceMetric<f64, Output = f64>,
    {
        let (strat, terminal_base_idx, query_value) = self.descend_complete_blocks(query);
        let terminal_rank =
            compare_terminal_block2_scalar(&self.stem_words, terminal_base_idx, query_value);
        debug_assert!(strat.leaf_idx_with_terminal_rank(terminal_rank) < self.leaf_count);
        let descriptor = unsafe {
            *self
                .stem_words
                .get_unchecked(terminal_base_idx + 3 + usize::from(terminal_rank))
        };
        let (byte_offset, leaf_len) = unpack_descriptor(descriptor);
        let arena = self
            .leaves
            .leaf_arena_from_embedded_descriptor(byte_offset, leaf_len);
        process_arena::<D, K>(&arena, query)
    }

    /// Same hybrid traversal followed by the existing extent-table lookup.
    #[inline(always)]
    pub fn approx_nearest_one_via_extents_shallow_simd<D>(&self, query: &[f64; K]) -> (f64, u32)
    where
        D: DistanceMetric<f64, Output = f64>,
    {
        let (strat, terminal_base_idx, query_value) = self.descend_complete_blocks(query);
        let terminal_rank =
            compare_terminal_block2_f64(&self.stem_words, terminal_base_idx, query_value);
        let leaf_idx = strat.leaf_idx_with_terminal_rank(terminal_rank);
        debug_assert!(leaf_idx < self.leaf_count);
        let arena = <VecOfArenas<f64, u32, K, B> as LeafStrategy<
            f64,
            u32,
            DonnellySimdDescentLeafEmbedded3,
            K,
            B,
        >>::leaf_arena(&self.leaves, leaf_idx);
        process_arena::<D, K>(&arena, query)
    }

    /// Number of logical leaves, exposed for benchmark diagnostics.
    #[inline]
    pub fn leaf_count(&self) -> usize {
        self.leaf_count
    }

    /// Number of packed stem/descriptor words, exposed for benchmark diagnostics.
    #[inline]
    pub fn stem_word_count(&self) -> usize {
        self.stem_words.len()
    }

    #[inline(always)]
    fn descend_complete_blocks(
        &self,
        query: &[f64; K],
    ) -> (DonnellySimdDescentLeafEmbedded3, usize, f64) {
        let stems_ptr = NonNull::new(self.stem_words.as_ptr() as *mut u8).unwrap();
        let stems = unsafe {
            std::slice::from_raw_parts(
                self.stem_words.as_ptr().cast::<f64>(),
                self.stem_words.len(),
            )
        };
        let mut strat = DonnellySimdDescentLeafEmbedded3::new(stems_ptr);
        let pivot_depth = self.max_stem_level + 1;
        let terminal_block_level = pivot_depth - 2;

        debug_assert_eq!((pivot_depth + 1) % 3, 0);
        debug_assert_eq!(terminal_block_level % 3, 0);

        while strat.level() < terminal_block_level {
            let query_value = unsafe { *query.get_unchecked(strat.dim::<K>()) };
            let child_idx = compare_block3(stems, query_value, strat.stem_idx());
            strat.traverse_block::<K>(child_idx);
        }

        debug_assert_eq!(strat.level(), terminal_block_level);
        let terminal_base_idx = strat.stem_idx();
        let query_value = unsafe { *query.get_unchecked(strat.dim::<K>()) };
        (strat, terminal_base_idx, query_value)
    }
}

#[inline(always)]
fn compare_terminal_block2_scalar(stem_words: &[u64], base_idx: usize, query_value: f64) -> u8 {
    let root = f64::from_bits(unsafe { *stem_words.get_unchecked(base_idx) });
    let root_right = query_value >= root;
    let child_idx = base_idx + 1 + usize::from(root_right);
    let child = f64::from_bits(unsafe { *stem_words.get_unchecked(child_idx) });
    (u8::from(root_right) << 1) | u8::from(query_value >= child)
}

#[inline(always)]
fn compare_terminal_block2_f64(stem_words: &[u64], base_idx: usize, query_value: f64) -> u8 {
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        unsafe {
            use std::arch::x86_64::*;

            let ptr = stem_words.as_ptr().add(base_idx).cast::<f64>();
            let pivots = _mm256_loadu_pd(ptr);
            let query = _mm256_set1_pd(query_value);
            let compared = _mm256_cmp_pd(query, pivots, _CMP_GE_OQ);
            return ((_mm256_movemask_pd(compared) as u32) & 0b0111).count_ones() as u8;
        }
    }

    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    {
        unsafe {
            use std::arch::aarch64::*;

            let ptr = stem_words.as_ptr().add(base_idx).cast::<f64>();
            let query = vdupq_n_f64(query_value);
            let first_two = vshrq_n_u64(vcgeq_f64(query, vld1q_f64(ptr)), 63);
            let third_and_descriptor = vshrq_n_u64(vcgeq_f64(query, vld1q_f64(ptr.add(2))), 63);
            return (vaddvq_u64(first_two) + vgetq_lane_u64::<0>(third_and_descriptor)) as u8;
        }
    }

    #[cfg(not(any(
        all(feature = "simd", target_arch = "x86_64"),
        all(feature = "simd", target_arch = "aarch64")
    )))]
    {
        let mut rank = 0u8;
        for offset in 0..3 {
            let pivot = f64::from_bits(unsafe { *stem_words.get_unchecked(base_idx + offset) });
            rank += u8::from(query_value >= pivot);
        }
        rank
    }
}

#[inline(always)]
fn process_arena<D, const K: usize>(
    arena: &crate::leaf_view::LeafArena<'_, f64, u32, K>,
    query: &[f64; K],
) -> (f64, u32)
where
    D: DistanceMetric<f64, Output = f64>,
{
    let mut best_dist = f64::INFINITY;
    let mut best_item = 0;
    crate::leaf_view_chunked::nearest_one::nearest_one_with_query_wide_arena::<f64, u32, D, K>(
        arena,
        query,
        &mut best_dist,
        &mut best_item,
    );
    (best_dist, best_item)
}

fn pack_descriptor(
    leaf_idx: usize,
    byte_offset: usize,
    leaf_len: usize,
) -> Result<u64, EmbeddedLeafDescriptorError> {
    if byte_offset as u128 > DESCRIPTOR_OFFSET_MASK as u128 {
        return Err(EmbeddedLeafDescriptorError::ByteOffsetOverflow {
            leaf_idx,
            byte_offset,
        });
    }
    let leaf_len = u16::try_from(leaf_len)
        .map_err(|_| EmbeddedLeafDescriptorError::LeafLengthOverflow { leaf_idx, leaf_len })?;
    Ok((u64::from(leaf_len) << DESCRIPTOR_OFFSET_BITS) | byte_offset as u64)
}

#[inline(always)]
fn unpack_descriptor(descriptor: u64) -> (usize, usize) {
    (
        (descriptor & DESCRIPTOR_OFFSET_MASK) as usize,
        (descriptor >> DESCRIPTOR_OFFSET_BITS) as usize,
    )
}

fn terminal_stem_idx<const K: usize>(leaf_idx: usize, pivot_depth: usize) -> usize {
    let mut strat = DonnellyUnrolledLeafEmbedded3::new_no_ptr();
    for bit_idx in (0..pivot_depth).rev() {
        let is_right = leaf_idx & (1usize << bit_idx) != 0;
        strat.traverse::<f64, K>(is_right);
    }
    strat.stem_idx()
}

fn terminal_simd_stem_idx<const K: usize>(leaf_idx: usize, pivot_depth: usize) -> usize {
    let mut strat = DonnellySimdDescentLeafEmbedded3::new_no_ptr();
    for bit_idx in (0..pivot_depth).rev() {
        let is_right = leaf_idx & (1usize << bit_idx) != 0;
        strat.traverse::<f64, K>(is_right);
    }
    strat.stem_idx()
}

#[cfg(test)]
mod tests {
    use super::{EmbeddedLeafDescriptorF64Tree, EmbeddedLeafDescriptorSimdF64Tree};
    use crate::dist::SquaredEuclidean;
    use rand::{RngExt, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn embedded_and_extent_paths_match() {
        const K: usize = 3;
        const B: usize = 32;
        let mut rng = ChaCha8Rng::seed_from_u64(0x5eed_edd0);
        let points: Vec<[f64; K]> = (0..4096).map(|_| rng.random()).collect();
        let queries: Vec<[f64; K]> = (0..256).map(|_| rng.random()).collect();
        let tree = EmbeddedLeafDescriptorF64Tree::<K, B>::new_from_slice(&points).unwrap();

        assert_eq!(tree.leaf_count(), 128);
        assert!(tree.stem_word_count() > tree.leaf_count());
        for query in &queries {
            assert_eq!(
                tree.approx_nearest_one_via_extents::<SquaredEuclidean<f64>>(query),
                tree.approx_nearest_one_embedded::<SquaredEuclidean<f64>>(query)
            );
        }
    }

    #[test]
    fn hybrid_embedded_scalar_and_extent_paths_match() {
        const K: usize = 3;
        const B: usize = 32;
        let mut rng = ChaCha8Rng::seed_from_u64(0x5eed_51d0);
        let points: Vec<[f64; K]> = (0..4096).map(|_| rng.random()).collect();
        let queries: Vec<[f64; K]> = (0..256).map(|_| rng.random()).collect();
        let tree = EmbeddedLeafDescriptorSimdF64Tree::<K, B>::new_from_slice(&points).unwrap();

        assert_eq!(tree.leaf_count(), 128);
        assert!(tree.stem_word_count() > tree.leaf_count());
        for query in &queries {
            let shallow =
                tree.approx_nearest_one_embedded_shallow_simd::<SquaredEuclidean<f64>>(query);
            assert_eq!(
                shallow,
                tree.approx_nearest_one_embedded_terminal_scalar::<SquaredEuclidean<f64>>(query)
            );
            assert_eq!(
                shallow,
                tree.approx_nearest_one_via_extents_shallow_simd::<SquaredEuclidean<f64>>(query)
            );
        }
    }
}
